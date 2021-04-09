import warnings

import numpy as np
import pandas as pd
import torch
import copy
import sdmetrics
from table_evaluator.table_evaluator import TableEvaluator
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer
from sdv.metrics.tabular import *

class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pack=1):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=1, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_penalty = ((
            gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        ) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


class CTGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=0, batch_size=500, discriminator_steps=1, log_frequency=True,
                 verbose=False, epochs=300, external_eval_target=False, adaptive_training=True):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._external_eval = False if external_eval_target==False else {
        "best_score": -np.inf,
        "correlation_scores": [],
        "detection_scores": [],
        "ml_efficacy_scores": {"tree":[], "adaboost":[], "regression":[], "mlp":[], "ml_efficacy":[]},
        "target":external_eval_target
        }
        self._adaptive_training = False if adaptive_training == False else {
            "r_d": np.random.random(),
            "r_g": np.random.random(),
            "prev_loss_g": np.random.random(),
            "prev_loss_d": np.random.random(),
            "lambda": 1/3,
        }

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))

    def fit(self, train_data, discrete_columns=tuple(), epochs=None, metadata_top_layer=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        #self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        # Data structures for the intermediate eval function
        original_training_data = train_data.copy()
        self._external_eval["eval_size"] = min(len(train_data), 10000)

        train_data = self._transformer.transform(train_data)
        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim
        ).to(self._device)

        self._optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        self._optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                if self._adaptive_training:
                    loss_g = self.calc_loss_g(mean, std)
                    pen, loss_d = self.calc_loss_d(mean, std)
                    if (self._adaptive_training["r_d"] >= (self._adaptive_training["lambda"] * self._adaptive_training["r_g"])):
                        self._optimizerD.zero_grad()
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        self._optimizerD.step()
                    else:
                        self._optimizerG.zero_grad()
                        loss_g.backward()
                        self._optimizerG.step()

                    loss_d = loss_d.detach().item()
                    loss_g = loss_g.detach().item()
                    self._adaptive_training["r_d"] = np.abs((loss_d - self._adaptive_training["prev_loss_d"]) / self._adaptive_training["prev_loss_d"])
                    self._adaptive_training["r_g"] = np.abs((loss_g - self._adaptive_training["prev_loss_g"])/ self._adaptive_training["prev_loss_g"])
                    self._adaptive_training["prev_loss_g"]= loss_g
                    self._adaptive_training["prev_loss_d"] = loss_d
                else:
                    for n in range(self._discriminator_steps):
                        self._optimizerD.zero_grad()
                        pen.backward(retain_graph=True)
                        loss_d.backward()
                        self._optimizerD.step()

                    self._optimizerG.zero_grad()
                    loss_g.backward()
                    self._optimizerG.step()

            if self._verbose:
                print("Epoch " + str(i+1))
                pass
            
            if self._external_eval != False:
                if i % 10 == 0:
                    # Reverse data back to its original format to compute external eval scores
                    real_data = original_training_data.sample(self._external_eval["eval_size"]).reset_index()
                    real_data = metadata_top_layer.reverse_transform(real_data)
                    synthetic_data = self.sample(self._external_eval["eval_size"])
                    synthetic_data = metadata_top_layer.reverse_transform(synthetic_data)
                    self.evaluate(synthetic_data, real_data, i+1)

        if self._external_eval != False:
            self._generator = self._external_eval["best_generator"]

    def calc_loss_d(self, mean, std):
        fakez = torch.normal(mean=mean, std=std)

        condvec = self._data_sampler.sample_condvec(self._batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = self._data_sampler.sample_data(self._batch_size, col, opt)
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            m1 = torch.from_numpy(m1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)

            perm = np.arange(self._batch_size)
            np.random.shuffle(perm)
            real = self._data_sampler.sample_data(
                self._batch_size, col[perm], opt[perm])
            c2 = c1[perm]

        fake = self._generator(fakez)
        fakeact = self._apply_activate(fake)

        real = torch.from_numpy(real.astype('float32')).to(self._device)

        if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real, c2], dim=1)
        else:
            real_cat = real
            fake_cat = fake

        y_fake = self._discriminator(fake_cat)
        y_real = self._discriminator(real_cat)

        pen = self._discriminator.calc_gradient_penalty(
            real_cat, fake_cat, self._device)
        loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

        return pen, loss_d

    def calc_loss_g(self, mean, std):
        fakez = torch.normal(mean=mean, std=std)
        condvec = self._data_sampler.sample_condvec(self._batch_size)

        if condvec is None:
            c1, m1, col, opt = None, None, None, None
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self._device)
            m1 = torch.from_numpy(m1).to(self._device)
            fakez = torch.cat([fakez, c1], dim=1)

        fake = self._generator(fakez)
        fakeact = self._apply_activate(fake)

        if c1 is not None:
            y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
        else:
            y_fake = self._discriminator(fakeact)

        if condvec is None:
            cross_entropy = 0
        else:
            cross_entropy = self._cond_loss(fake, c1, m1)

        loss_g = -torch.mean(y_fake) + cross_entropy

        return loss_g

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def evaluate(self, synthetic_data, real_data, epoch):
        categorical_cols = real_data.select_dtypes("object").columns.values.tolist()

        correlation_score = self.compute_correlation_score(synthetic_data, real_data, categorical_cols)
        self._external_eval["correlation_scores"].append(correlation_score)

        ml_efficacy_score = self.compute_ml_efficacy(real_data, synthetic_data, self._external_eval["target"])
        
        detection_score = self.compute_detection(real_data, synthetic_data)
        self._external_eval["detection_scores"].append(detection_score)

        overall_score = (detection_score*0.5+ml_efficacy_score*0.5)

        if self._external_eval["best_score"] < overall_score:
            print("New max score!")
            print(str(overall_score))
            self._external_eval["best_model_epoch"] = epoch
            self._external_eval["best_score"] = overall_score
            self._external_eval["best_generator"] = copy.deepcopy(self._generator)

    def get_problem_type(self, data, target):
        target_column = data[target]

        if target_column.dtypes == "object":
            unique_labels = np.unique(target_column)

            if len(unique_labels) == 2:
                problem_type = "binary_classification"

            else:
                problem_type = "multi_classification"
        else:
            raise AttributeError("Regression ml efficacy not yet implemented") 

        return problem_type

    def compute_detection(self, real_data, synthetic_data, verbose=True):
        real_data = real_data.dropna()
        synthetic_data = synthetic_data.dropna()
        detection_score = LogisticDetection.compute(real_data, synthetic_data)

        if verbose:
            print("Detection score: " + str(detection_score))

        return detection_score

    def compute_ml_efficacy(self, real_data, synthetic_data, target, verbose = True):
        dtypes = real_data.dtypes.tolist()
        problem_type = self.get_problem_type(real_data, target)
        scores = []

        if problem_type == "binary_classification":
            (tree_score, _), _ = BinaryDecisionTreeClassifier.compute(real_data, synthetic_data, dtypes=dtypes.copy(), target=target)
            (adaboost_score, _), _ = BinaryAdaBoostClassifier.compute(real_data, synthetic_data, dtypes=dtypes.copy(), target=target)
            (regression_score, _), _ = BinaryLogisticRegression.compute(real_data, synthetic_data, dtypes=dtypes.copy(), target=target)
            (mlp_score, _), _ = BinaryMLPClassifier.compute(real_data, synthetic_data, dtypes=dtypes.copy(), target=target)

            scores.extend([tree_score, adaboost_score, regression_score, mlp_score])
            ml_efficacy_score = sum(scores)/len(scores)

            self._external_eval["ml_efficacy_scores"]["tree"].append(tree_score)
            self._external_eval["ml_efficacy_scores"]["adaboost"].append(adaboost_score)
            self._external_eval["ml_efficacy_scores"]["regression"].append(regression_score)
            self._external_eval["ml_efficacy_scores"]["mlp"].append(mlp_score)
            self._external_eval["ml_efficacy_scores"]["ml_efficacy"].append(ml_efficacy_score)

            if verbose:
                print("Tree score: " + str(tree_score))
                print("Adaboost score: " + str(adaboost_score))
                print("Regression score: " + str(regression_score))
                print("Mlp score: " + str(mlp_score))
                print("ML efficay score: " + str(ml_efficacy_score))

        elif problem_type == "multi_classification":
            tree_score, _ = MulticlassDecisionTreeClassifier.compute(real_data, synthetic_data, dtypes=dtypes.copy(), target=target)
            mlp_score, _ = MulticlassMLPClassifier.compute(real_data, synthetic_data, dtypes=dtypes.copy(), target=target)

            scores.extend([tree_score, mlp_score])
            ml_efficacy_score = sum(scores)/len(scores)

            self._external_eval["ml_efficacy_scores"]["tree"].append(tree_score)
            self._external_eval["ml_efficacy_scores"]["mlp"].append(mlp_score)
            self._external_eval["ml_efficacy_scores"]["ml_efficacy"].append(ml_efficacy_score)

            if verbose:
                print("Tree score: " + str(tree_score))
                print("Mlp score: " + str(mlp_score))
                print("ML efficay score: " + str(ml_efficacy_score))
        
        return ml_efficacy_score

    def compute_correlation_score(self, synthetic_data, real_data, categorical_cols, verbose=True):
        table_eval = TableEvaluator(real_data, synthetic_data, cat_cols=categorical_cols, verbose=False)
        correlation_score = table_eval.correlation_distance(how='rmse')

        if verbose:
            print("Rmse correlation: " + str(correlation_score))

        return correlation_score

    def get_metadata(self):
        meta_data = {}
        data_info = {}
        categorical_cols = []
        dtypes = []
        dtypes_mapping = {}
        for index, column in enumerate(self._transformer._column_transform_info_list):
            name = column[0]
            data_type = column[1]
            if data_type == "discrete":
                data_info[name] = {"type": "categorical"}
                categorical_cols.append(name)
                dtypes.append("object")
                dtypes_mapping[name] = "object"
            else:
                if self._transformer._column_raw_dtypes._selected_obj[index] == "int64":
                    data_info[name] = {"type": "numerical", "subtype": "integer"}
                    dtypes.append(int)
                    dtypes_mapping[name] = int
                else:
                    data_info[name] = {"type": "numerical", "subtype": "float"}
                    dtypes.append(np.float64)
                    dtypes_mapping[name] = np.float64
        meta_data["tables"] = {None: {"fields" : data_info}}

        return meta_data, categorical_cols, dtypes, dtypes_mapping

    def set_device(self, device):
        self._device = device
        if hasattr(self, '_generator'):
            self._generator.to(self._device)
        if hasattr(self, '_discriminator'):
            self._discriminator.to(self._device)
