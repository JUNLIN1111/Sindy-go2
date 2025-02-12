import warnings
warnings.filterwarnings('ignore')
import logging
import os
import numpy as np

from ray.rllib.algorithms.registry import ALGORITHMS as rllib_algos
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune, air
from ray.air import session, Checkpoint

from sindy_rl.dyna import DynaSINDy

def dyna_sindy(config):
    '''
    ray.Tune 的函数式 API,于定义实验
    '''
    dyna_config = config
    train_iterations = config['n_train_iter']
    dyn_fit_freq = config['dyn_fit_freq']
    ckpt_freq = config['ray_config']['checkpoint_freq']
    
    # 初始化时收集（或加载）数据
    dyna = DynaSINDy(dyna_config)
    
    start_iter = 0
    
    # 对于 PBT，在评估种群并修剪掉表现最差的个体后，session 会包含一个检查点
    checkpoint = session.get_checkpoint()
    if checkpoint:
        check_dict = checkpoint.to_dict()
        dyna.load_checkpoint(check_dict)
        
        # 获取迭代次数，以确保正确进行检查点保存
        start_iter = check_dict['epoch'] + 1

    # 设置动力学、奖励、DRL 算法，将权重推送到代理模型
    # 在远程工作者上执行
    dyna.fit_dynamics()
    dyna.fit_rew()
    dyna.update_surrogate()

    collect_dict = {'mean_rew': np.nan, 
                    'mean_len': 0}
    
    # 主训练循环
    for n_iter in range(start_iter, train_iterations):
        checkpoint = None
        train_results = dyna.train_algo()

        # 定期通过收集 on-policy 数据进行评估
        if (n_iter % dyn_fit_freq) == dyn_fit_freq - 1:
            (trajs_obs, 
             trajs_acts, 
             trajs_rew) = dyna.collect_data(dyna.on_policy_buffer,
                                            dyna.real_env,
                                            dyna.on_policy_pi,
                                            **dyna_config['on_policy_buffer']['collect']
                                            )
            dyna.fit_dynamics()
            dyna.fit_rew()
            dyna.update_surrogate()
            
            collect_dict = {}
            collect_dict['mean_rew'] = np.mean([np.sum(rew) for rew in trajs_rew])
            collect_dict['mean_len'] = np.mean([len(obs) for obs in trajs_obs]) 
            
        # 检查点保存（理想情况下在最新一次数据收集后进行）
        if ((n_iter % ckpt_freq) == ckpt_freq - 1):
            
            check_dict = dyna.save_checkpoint(ckpt_num=n_iter, 
                                              save_dir = session.get_trial_dir(),
                                                )
            checkpoint = Checkpoint.from_dict(check_dict)
        
        # 编译指标供 tune 报告
        train_results['traj_buffer'] = dyna.get_buffer_metrics()
        train_results['dyn_collect'] = collect_dict
        
        # 可能不完全必要，来源于一些官方的 Tune PBT 示例
        if checkpoint:
            session.report(train_results, 
                           checkpoint=checkpoint)
        else:
            session.report(train_results)
    
def explore(dyna_config):
    '''
    用于 PBT。
    确保探索后的（连续）参数保持在给定范围内
    '''
    config = dyna_config['drl']['config']['training']
    
    config['lambda_'] = np.clip(config['lambda_'], 0, 1)
    config['gamma'] = np.clip(config['gamma'], 0, 1)
    
    dyna_config['drl']['config']['training'] = config
    return dyna_config

if __name__ == '__main__': 
    import yaml
    import logging
    import ray
    
    from pprint import pprint
    
    from sindy_rl.policy import RandomPolicy
    from sindy_rl import _parent_dir
    
    # 待办：替换为 argparse
    filename = os.path.join(_parent_dir, 
                            'sindy_rl',
                            'config_templates', 
                            'go.yml' # 替换为适当的配置 yml 文件
                            )
    
    # 加载配置文件
    with open(filename, 'r') as f:
        dyna_config = yaml.load(f, Loader=yaml.SafeLoader)
    LOCAL_DIR =  os.path.join(_parent_dir, 'ray_results',dyna_config['exp_dir'])
    pprint(dyna_config)
    
    # 设置日志记录器
    logging.basicConfig()
    logger = logging.getLogger('dyna-sindy')
    logger.setLevel(logging.INFO)

    # 初始化默认的 off-policy，用于初始数据收集
    n_control = dyna_config['drl']['config']['environment']['env_config']['act_dim']
    dyna_config['off_policy_pi'] = RandomPolicy(low=-1*np.ones(n_control), 
                                                high = np.ones(n_control), 
                                                seed=0)
    
    # 设置 ray
    ip_head = os.environ.get('ip_head', None)
    ray.init(address=ip_head, 
             logging_level=logging.ERROR)

    # PBT 专用配置,这里没用
    pbt_sched = None
    if dyna_config.get('use_pbt', False):
        pbt_config = dyna_config['pbt_config']
        hyper_mut = {}
        for key, val in pbt_config['hyperparam_mutations'].items():
            search_class = getattr(tune, val['search_class'])
            hyper_mut[key] = search_class(*val['search_space'])
        
        pbt_config['hyperparam_mutations'] = hyper_mut
            
        pbt_sched = PopulationBasedTraining(
                        **pbt_config,
                        custom_explore_fn=explore
                    )

    # ray + tune 配置
    ray_config = dyna_config['ray_config']
    run_config=air.RunConfig(
        local_dir=LOCAL_DIR,
        **ray_config['run_config']
    )
    
    tune_config=tune.TuneConfig(
                    **dyna_config['ray_config']['tune_config'],
                    scheduler=pbt_sched
                    )
    
    drl_class, drl_default_config = rllib_algos.get(dyna_config['drl']['class'])()
    
    tune.Tuner(
        tune.with_resources(dyna_sindy, 
                            drl_class.default_resource_request(drl_default_config)
                            ),
        param_space=dyna_config, # 这是传递给实验的配置
        run_config=run_config,
        tune_config=tune_config
    ).fit()