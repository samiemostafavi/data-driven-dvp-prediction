import numpy as np
import tensorflow as tf

import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.tf_utils.network import MLP
from cde.utils.tf_utils.adamW import AdamWOptimizer
from .BaseNNEstimator import BaseNNEstimator
from cde.utils.serializable import Serializable


class PowerLawTailDistribution():
    """ Power Law Tail Distribution
        Args:

    """
    def __init__(self, tail_boundry, tail_param):

        # constrain u before assigning it
        self.tail_boundry = tf.convert_to_tensor(tail_boundry,dtype=tf.float32)
        self.tail_param = tail_param

    def test_prob(self, values):
        xmin = self.tail_boundry
        alpha = tf.pow(self.tail_param,-1)
        res = tf.pow(tf.divide(values,xmin),-alpha)
        res = tf.squeeze(res)
        return values, xmin, alpha, res

    def prob(self, values):
        xmin = self.tail_boundry
        alpha = tf.pow(self.tail_param,-1)
        res = tf.pow(tf.divide(values,xmin),-alpha)
        return tf.squeeze(res,axis=1)

    def log_prob(self, values):
        return tf.log(self.prob(values=values))

    def test_like(self, values):
        xmin = self.tail_boundry
        alpha = tf.pow(self.tail_param,-1)
        #greater = tf.squeeze(tf.greater(values,xmin),axis=1)
        #indices = tf.where(greater)
        #newx = tf.reshape(tf.gather(values, indices),[-1])
        #newalpha = tf.reshape(tf.gather(alpha, indices),[-1])
        newx = values
        newalpha = alpha
        elems = tf.log(newalpha-1)-tf.log(xmin)-newalpha*(tf.log(tf.divide(newx,xmin)))
        res = tf.reduce_sum(elems)
        #return values,xmin,alpha,greater,indices,newx,newalpha,elems,res
        return values,xmin,alpha,newx,newalpha,elems,res

    def log_like(self, values):
        xmin = self.tail_boundry
        alpha = tf.pow(self.tail_param,-1)
        #greater = tf.squeeze(tf.greater(values,xmin),axis=1)
        #indices = tf.where(greater)
        #newx = tf.reshape(tf.gather(values, indices),[-1])
        #newalpha = tf.reshape(tf.gather(alpha, indices),[-1])
        newx = values
        newalpha = alpha
        elems = tf.log(alpha-1)-tf.log(xmin)-alpha*(tf.log(tf.divide(newx,xmin)))
        return tf.reduce_sum(elems)

class TailIndexEstimator(BaseNNEstimator):
    """ Tail Index Estimator

        Args:
            name: (str) name space of the network (should be unique in code, otherwise tensorflow namespace collisions may arise)
            ndim_x: (int) dimensionality of x variable
            tail_boundries_y: (tuple of int) starting points of right tail region
            hidden_sizes: (tuple of int) sizes of the hidden layers of the neural network
            hidden_nonlinearity: (tf function) nonlinearity of the hidden layers
            n_training_epochs: (int) Number of epochs for training
            weight_decay: (float) the amount of decoupled (http://arxiv.org/abs/1711.05101) weight decay to apply
            weight_normalization: (boolean) whether weight normalization shall be used for the neural network
            data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and uniform-std
            dropout: (float) the probability of switching off nodes during training
            random_seed: (optional) seed (int) of the random number generators used
    """

    def __init__(self, name, ndim_x, tail_boundries_y=[3,4,5,6,7,8], hidden_sizes=(16, 16),
                 hidden_nonlinearity=tf.tanh, n_training_epochs=1000,
                 weight_decay=0.0, weight_normalization=False, data_normalization=False, dropout=0.0,
                 random_seed=None,verbose_step=100,learning_rate=5e-3):
        Serializable.quick_init(self, locals())
        self._check_uniqueness_of_scope(name)

        self.verbose_step = verbose_step
        self.learning_rate = learning_rate

        self.name = name
        self.ndim_x = ndim_x
        self.ndim_y = 1
        self.tail_boundries_y = tail_boundries_y

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(seed=random_seed)
        tf.set_random_seed(random_seed)

        # specification of the network
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

        self.n_training_epochs = n_training_epochs

        # decoupled weight decay
        self.weight_decay = weight_decay

        # normalizing the network weights
        self.weight_normalization = weight_normalization

        # whether to normalize the data to zero mean, and uniform variance
        self.data_normalization = data_normalization

        # the prob of dropping a node
        self.dropout = dropout

        # sampling is not yet supported
        self.can_sample = False
        self.has_pdf = True
        self.has_cdf = False

        # regularization parameters
        self.x_noise_std = None
        self.y_noise_std = None
        self.adaptive_noise_fn = None

        self.fitted = False

        # build tensorflow model
        self._build_model()

    def starttfsession(self):
        # If no session has yet been created, create one and make it the default
        self.sess = tf.get_default_session() if tf.get_default_session() else tf.InteractiveSession()

    def _setup_inference_and_initialize(self):
        # If no session has yet been created, create one and make it the default
        self.sess = tf.get_default_session() if tf.get_default_session() else tf.InteractiveSession()

    def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
        """
        Fit the model with to the provided data

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        :param eval_set: (tuple) eval/test dataset - tuple (X_test, Y_test)
        :param verbose: (boolean) controls the verbosity of console output
        """  

        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

        if eval_set:
            eval_set = tuple(self._handle_input_dimensionality(x) for x in eval_set)

        # If no session has yet been created, create one and make it the default
        self.sess = tf.get_default_session() if tf.get_default_session() else tf.InteractiveSession()

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        tf.initializers.variables(var_list, name='init').run()

        if self.data_normalization:
            self._compute_data_normalization(X, Y)

        self._compute_noise_intensity(X, Y)

        for i in range(0, self.n_training_epochs + 1):
            for j in range(self.mlp_size):
                
                self.N[j] = self.N[j] + len(X)
                xmin = self.tail_boundries_y[j]
                aw = np.squeeze(np.argwhere(np.squeeze(Y)>=xmin))
                newY = Y[aw]
                newX = np.squeeze(X[aw])
                self.K[j] = self.K[j] + len(newX)
                if(len(newX) > 0):
                    self.sess.run(self.train_steps[j],
                                feed_dict={self.X_ph: newX, self.Y_ph: newY, self.train_phase: True, self.dropout_ph: self.dropout})

                    self.aics[j] = self.sess.run(self.calc_aics[j],
                                        feed_dict={self.X_ph: newX, self.Y_ph: newY, self.train_phase: False, self.dropout_ph: self.dropout})

                    self.lls[j] = self.sess.run(self.calc_lls[j],
                                        feed_dict={self.X_ph: newX, self.Y_ph: newY, self.train_phase: False, self.dropout_ph: self.dropout})
                    
                    
                    #ll = self.sess.run(self.prob_loggers, feed_dict={self.X_ph: X, self.Y_ph: Y, self.K_ph: K, self.N_ph: N})
                    #print(ll)

            if verbose and not i % self.verbose_step:
                    print("Step " + str(i) + ": train log-likelihood " + str(self.lls) + " aics " + str(self.aics))

                
                    
            

        self.fitted = True

    def check(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
        
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)

        # If no session has yet been created, create one and make it the default
        self.sess = tf.get_default_session() if tf.get_default_session() else tf.InteractiveSession()

        for j in range(self.mlp_size):
            print(self.K[j])
            print(self.N[j])
            ll = self.sess.run(self.prob_loggers, feed_dict={self.X_ph: X, self.Y_ph: Y}) #, 
            print(ll)

            #print(X)
            #print(Y)
            #ll = self.sess.run(self.prob_loggers, feed_dict={self.X_ph: X, self.Y_ph: Y})
            #print(ll)

    def tail_prob(self, X, Y):
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p*self.K[0]/self.N[0]

    def reset_fit(self):
        """
        Resets all tensorflow objects and enables this model to be fitted anew
        """
        tf.reset_default_graph()
        self._build_model()
        self.fitted = False

    def _param_grid(self):
        return {
            'n_training_epochs': [500, 1000, 1500],
            'hidden_sizes': [(16, 16), (32, 32)],
            'tail_boundries_y': [
                (3,4,5,6,7,8),
                (1,2,3,4,5,6,7,8,9,10),
                (10,11,12,13,14,15,16,17,18,19,20),
                (20,21,22,23,24,25,26,27,28,29,30),
            ],
            'weight_decay': [1e-5, 0.0]
        }

    def _build_model(self):
        """
        implementation of the tail estimator model
        """
        with tf.variable_scope(self.name):
            # adds placeholders, data normalization to graph as desired. Also sets up a placeholder
            # for dropout
            self.layer_in_x, self.layer_in_y = self._build_input_layers()
            self.y_input = L.get_output(self.layer_in_y)

            # get the number of tail params
            param_split_sizes = np.ones(len(self.tail_boundries_y),dtype=np.int32)
            self.mlp_size = len(self.tail_boundries_y)

            self.N = np.zeros(self.mlp_size)
            self.K = np.zeros(self.mlp_size)
            self.aics = np.zeros(self.mlp_size)
            self.lls = np.zeros(self.mlp_size)

            self.core_networks = []
            self.tail_params = []
            self.tail_dists = []
            self.pdfs = []
            self.log_pdfs = []
            self.log_likes = []
            self.loggers = []
            self.prob_loggers = []
            self.log_losses = []
            self.train_steps = []
            self.calc_aics = []
            self.calc_lls = []
            for i in range(self.mlp_size):
                core_network = MLP(
                    name="core_network_"+str(i),
                    input_layer=self.layer_in_x,
                    output_dim=1,
                    hidden_sizes=self.hidden_sizes,
                    hidden_nonlinearity=self.hidden_nonlinearity,
                    output_nonlinearity=tf.sigmoid,
                    weight_normalization=self.weight_normalization,
                    dropout_ph=self.dropout_ph if self.dropout else None
                )
                self.core_networks.append(core_network)
                tail_param = L.get_output(core_network.output_layer)
                self.tail_params.append(tail_param)

                tail_boundry = self.tail_boundries_y[i]
                # instanciate the tail distributions with their parameters
                tail_dist = PowerLawTailDistribution(tail_boundry=tail_boundry,tail_param=tail_param)
                self.tail_dists.append(tail_dist)

                # since we operate with matrices not vectors, the output would have dimension (?,1)
                # and therefor has to be reduce first to have shape (?,)
                # for x shape (batch_size, 1) normal_distribution.pdf(x) outputs shape (batch_size, 1) -> squeeze
                pdf = tail_dist.prob(values=self.y_input)
                log_pdf = tail_dist.log_prob(values=self.y_input)
                if self.data_normalization:
                    pdf = pdf / tf.reduce_prod(self.std_y_sym)
                    log_pdf = log_pdf - tf.reduce_sum(tf.log(self.std_y_sym))

                log_like = tail_dist.log_like(self.y_input)
                log_loss = -log_like
                self.pdfs.append(pdf)
                self.log_pdfs.append(log_pdf)
                self.log_likes.append(log_like)
                self.loggers.append(tail_dist.test_like(self.y_input))
                self.prob_loggers.append(tail_dist.test_prob(values=self.y_input))
                self.log_losses.append(log_loss)
                self.calc_aics.append(2-2*log_like)
                self.calc_lls.append(log_like)

                optimizer = AdamWOptimizer(self.weight_decay, learning_rate=self.learning_rate) if self.weight_decay else tf.train.AdamOptimizer()
                self.train_steps.append(optimizer.minimize(log_loss))
            
            self.log_loss = tf.reduce_sum(self.log_losses)
            self.pdf_ = self.pdfs[0]
            self.log_pdf_ = self.log_pdfs[0]

        # initialize LayersPowered -> provides functions for serializing tf models
        #LayersPowered.__init__(self, [self.layer_in_y, core_network.output_layer])

    def __str__(self):
        return "\nEstimator type: {}" \
               "\n tail_boundries_y: {}" \
               "\n data_normalization: {}" \
               "\n weight_normalization: {}" \
               "\n n_training_epochs: {}" \
               "\n ".format(self.__class__.__name__, self.tail_boundries_y, self.data_normalization,
                            self.weight_normalization, self.n_training_epochs)
