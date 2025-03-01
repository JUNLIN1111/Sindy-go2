import numpy as np
import pysindy as ps
def lin_and_cube_library(poly_int = False):
    '''
    将多项式库限制为线性项和三次项的组合。

    参数：
        `poly_int` (布尔值): 是否包含多项式（三次）交互项（例如 x^2 y）。
            如果为 False，则仅包含齐次项（例如 x^3, y^3）。
    '''
    if poly_int:
        # 如果包含交互项，定义库函数和名称
        library_functions = [
            lambda x: x,  # 线性项
            lambda x,y: x * y**2,  # 交互项（如 x y^2）
            lambda x: x**3  # 三次项
        ]
        library_names = [lambda x: x, 
                         lambda x,y: f'{x} {y}^2', 
                         lambda x: f'{x}^3'
                         ]
    else:
        # 如果不包含交互项，仅定义线性项和三次项
        library_functions = [
            lambda x: x,  # 线性项
            lambda x: x**3  # 三次项
        ]
        library_names = [lambda x: x, lambda x: f'{x}^3']

    # 创建自定义库
    polyLibCustom = ps.CustomLibrary(library_functions=library_functions, 
                                    function_names = library_names)

    return polyLibCustom


def get_affine_lib(poly_deg, n_state=2, n_control = 2, poly_int=False, tensor=False, use_cub_lin=False):
    '''
    创建控制仿射库，形式为：
        x_dot = p_1(x) + p_2(x)u
    其中 p_1 和 p_2 是关于状态变量 x 的多项式，阶数为 poly_deg。

    参数：
        `poly_deg`  (整数): 多项式 p_1 和 p_2 的阶数
        `n_state`   (整数): 状态变量 x 的维度
        `n_control` (整数): 控制变量 u 的维度
        `poly_int`  (布尔值): 是否包含多项式交互项
                    例如，当 poly_deg = 2 时，是否包含 x_1 * x_2 项，还是仅包含 (x_j)^2
        `tensor`    (布尔值): 是否包含 p_2（即多项式状态空间库与线性控制库的张量积）
        `use_cub_lin` (布尔值): 是否使用自定义的线性+三次项库
    '''
    if use_cub_lin:
        # 如果使用自定义的线性+三次项库，确保 poly_deg 为 3
        assert poly_deg==3, 'poly_deg 必须为 3 才能使用自定义的线性+三次项库'
        polyLib = lin_and_cube_library(poly_int = poly_int) 
    else:
        # 否则，使用标准的多项式库
        polyLib = ps.PolynomialLibrary(degree=poly_deg, 
                                        include_bias=False, 
                                        include_interaction=poly_int)
    
    # 创建线性控制库
    affineLib = ps.PolynomialLibrary(degree=1, 
                                    include_bias=False, 
                                    include_interaction=False)

    # 对于第一个库（状态库），不使用任何控制变量。
    # 通过将控制变量索引设置为 0，确保仅使用状态变量。
    inputs_state_library = np.arange(n_state + n_control)
    inputs_state_library[-n_control:] = 0
    
    # 对于第二个库（控制库），仅使用控制变量。
    # 通过将状态变量索引设置为 n_state + n_control - 1，确保仅使用控制变量。
    inputs_control_library = np.arange(n_state + n_control)
    inputs_control_library[:n_state] = n_state + n_control -1
    
    # 将两个库的输入索引组合成一个数组
    inputs_per_library = np.array([
                                    inputs_state_library,
                                    inputs_control_library
                                    ], dtype=int)

    # 如果 tensor 为 True，设置张量积数组
    if tensor:
        tensor_array = np.array([[1, 1]])
    else:
        tensor_array = None 

    # 创建广义库，将状态库和控制库组合起来
    generalized_library = ps.GeneralizedLibrary(
        [polyLib, affineLib],
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    return generalized_library

def get_affine_lib_from_base(base_lib, n_state=2, n_control = 2, include_bias = False):
    '''
    Returns library for regression of the form:
        W Phi(x) + W2 Phi(x) u
        
        `base_lib`  (ps.FeatureLibrary)
            base library, Phi
        `n_state`   (int):  The dimension of the state variable x
        `n_control` (int):  The dimension of the control variable u
    '''
    control_lib = ps.PolynomialLibrary(degree=1, 
                                    include_bias=False, 
                                    include_interaction=False)

    # For the first library, don't use any of the control variables.
    # forcing this to be zero just ensures that we're using the "zero-th"
    # indexed variable. The source code uses a `np.unique()` call
    inputs_state_library = np.arange(n_state + n_control)
    inputs_state_library[-n_control:] = 0
    
    # For the second library, we only want the control variables.
    #  forching the first `n_state` terms to be n_state + n_control - 1 is 
    #  just ensuring that we're using the last indexed variable (i.e. the
    #  last control term).
    inputs_control_library = np.arange(n_state + n_control)
    inputs_control_library[:n_state] = n_state + n_control -1
    
    inputs_per_library = np.array([
                                    inputs_state_library,
                                    inputs_control_library
                                    ], dtype=int)

    tensor_array = np.array([[1, 1]])

    libs = [base_lib, control_lib]
    if include_bias:
        libs = [ps.PolynomialLibrary(degree=0), base_lib, control_lib]
        tensor_array = np.array([[0, 1, 1,]])
        inputs_per_library = np.array([
                                    np.zeros(n_state + n_control),
                                    inputs_state_library,
                                    inputs_control_library,
                                    ], dtype=int)
    generalized_library = ps.GeneralizedLibrary(
        libs,
        tensor_array=tensor_array,
        inputs_per_library=inputs_per_library,
    )
    return generalized_library



def build_optimizer(config):
    '''
    Helper method to build a SINDy optimizer from a configuration dictionary
    
    Example Structure:
    'base_optimizer':
        'name': 'STLSQ'
        'kwargs': 
            'alpha': 1e-5
            'thresh': 1e-2
    'ensemble':
        'bagging': true
        'library_ensemble': true
        'n_models': 100
    '''
    opt_name = config['base_optimizer']['name']
    opt_kwargs = config['base_optimizer']['kwargs']
    base_optimizer = getattr(ps, opt_name)(**opt_kwargs)
    
    ensemble = config['ensemble']
    if ensemble:
        optimizer = ps.EnsembleOptimizer(opt=base_optimizer,
                                         **ensemble)
    else:
        optimizer =base_optimizer

    return optimizer

def build_feature_library(config):
    '''Build library from config'''
    # TO-DO: Make more general
    lib_name = config['name']
    lib_kwargs = config['kwargs']
    
    # custom affine library
    if lib_name == 'affine':
        lib = get_affine_lib(**lib_kwargs)
    else:
        lib_class = getattr(ps, lib_name)
        lib = lib_class(**lib_kwargs)
    return lib

#---------------------------------------------------
# Some spare code for taking strings and then building
# library functions from them.
#---------------------------------------------------
# import sympy
# from pysindy import CustomLibrary
# lib_names = ['{}', '{}^3', 'sin(2*{})', 'exp({})']
# lib_name_fns = [sym.format for sym in lib_names]

# var = sympy.var('x')
# lib_syms = [sympy.sympify(lib_fn('x')) for lib_fn in lib_name_fns]
# tmp_fn = [sympy.lambdify(var, sym, 'numpy') for sym in lib_syms]
# tmp = CustomLibrary(library_functions=tmp_fn, function_names=lib_name_fns)
# tmp.fit(np.ones(2))
# tmp.get_feature_names(['x', 'y'])


if __name__ == '__main__': 
    lib = get_affine_lib(poly_deg=2)
    