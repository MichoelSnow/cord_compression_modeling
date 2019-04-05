import keras
from cv_framework.metrics.metrics import muilticlass_logloss
import gin

@gin.configurable
def set_opt_params(optimizer, params):
    params_default_dict = {
        'SGD':{'lr':0.01, 'momentum':0.0, 'decay':0.0, 'nesterov':False},
        'RMSprop':{'lr':0.001, 'rho':0.9, 'epsilon':None, 'decay':0.0},
        'Adagrad':{'lr':0.01, 'epsilon':None, 'decay':0.0},
        'Adadelta':{'lr':1.0, 'rho':0.95, 'epsilon':None, 'decay':0.0},
        'Adam':{'lr':0.001, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'decay':0.0, 'amsgrad':False},
        'Adamax':{'lr':0.002, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'decay':0.0},
        'Nadam':{'lr':0.002, 'beta_1':0.9, 'beta_2':0.999, 'epsilon':None, 'schedule_decay':0.004},
    }

    allowed_opt_params = {
        'SGD':[
            'lr',
            'momentum',
            'decay',
            'nestrov'
        ],
        'RMSprop':[
            'lr',
            'rho',
            'epsilon',
            'decay'
        ],
        'Adagrad':[
            'lr',
            'epsilon',
            'decay'
        ],
        'Adadelta':[
            'lr',
            'rho',
            'epsilon',
            'decay'
        ],
        'Adam':[
            'lr',
            'beta_1',
            'beta_2',
            'epsilon',
            'decay',
            'amsgrad'
        ],
        'Adamax':[
            'lr',
            'beta_1',
            'beta_2',
            'epsilon',
            'decay'
        ],
        'Nadam':[
            'lr',
            'beta_1',
            'beta_2',
            'epsilon',
            'schedule_decay'
        ]
    }

    for k,v in params.items():
        if  all([True if x == None else False for x in v]):
            return params_default_dict[optimizer]
        else:
            for val in params:
                if params[val] not in allowed_opt_params[k]:
                    del params[val]
    return params

@gin.configurable
def set_optimizer(optimizer, lr=None, momentum=None, decay=None, nestrov=None, rho=None, epsilon=None,
                  beta_1=None, beta_2=None, amsgrad=None, schedule_decay=None):
    allowed_optimizers = [
        'SGD',
        'RMSprop',
        'Adagrad',
        'Adadelta',
        'Adam',
        'Adamax',
        'Nadam'
    ]

    params =  {
        'lr':lr,
        'momentum':momentum,
        'decay':decay,
        'nestrov':nestrov,
        'rho':rho,
        'epsilon':epsilon,
        'beta_1':beta_1,
        'beta_2':beta_2,
        'amsgrad':amsgrad,
        'schedule_decay':schedule_decay
    }

    if optimizer not in allowed_optimizers:
        raise ValueError(f'Optimzer {optimizer} must be in the allowed list: {allowed_optimizers}')

    fin_params = set_opt_params(optimizer, params)

    optimizers = {
        'SGD':keras.optimizers.SGD(**fin_params),
        'RMSprop':keras.optimizers.RMSprop(**fin_params),
        'Adagrad':keras.optimizers.Adagrad(**fin_params),
        'Adadelta':keras.optimizers.Adadelta(**fin_params),
        'Adam':keras.optimizers.Adam(**fin_params),
        'Adamax':keras.optimizers.adamax(**fin_params),
        'Nadam':keras.optimizers.Nadam(**fin_params),
        }

    return optimizers[optimizer]

@gin.configurable
def set_loss(loss='categorical_crossentropy'):
    allowed_losses = ['mean_squared_error',
                      'mean_absolute_error',
                      'mean_absolute_percentage_error',
                      'mean_squared_logarithmic_error',
                      'squared_hinge',
                      'hinge',
                      'categorical_hinge',
                      'logcosh',
                      'categorical_crossentropy',
                      'sparse_categorical_crossentropy',
                      'binary_crossentropy',
                      'kullback_leibler_divergence',
                      'poisson',
                      'cosine_proximity',
                      ]
    if loss not in allowed_losses:
        raise ValueError(f'Losses {loss} is not in allowed losses: {allowed_losses}')
    else:
        return loss

@gin.configurable
def set_batch_metrics(metrics=None):
    batch_metrics = ['acc']
    if not metrics:
        return batch_metrics
    elif isinstance(metrics, list):
        return batch_metrics + metrics
    else:
        raise ValueError('Batch metircs must be a list.')

@gin.configurable
def comp_model(model=None, **kwargs):
    k_opt = set_adam(**kwargs)
    k_loss = set_loss()
    k_metrics = set_batch_metrics(metrics=[muilticlass_logloss])
    model.compile(optimizer=k_opt, loss=k_loss, metrics=k_metrics)
