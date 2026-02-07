# borrow from https://github.com/liaohaofu/adn and add getEasyDict
__all__ = ["get_config", "update_config", "toYaml", "getEasyDict"]

import yaml


def get_config(config_file, config_names=[]):
    ''' load config from file
    '''

    with open(config_file) as f:
        # yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        
        yaml_dict = yaml.load(f, Loader=yaml.SafeLoader) ##PyYAML库的版本低于5.1

    return yaml_dict
##ondef

def dict2yaml(path, dic, listOneLine=True):
    if listOneLine:
        data_str = yaml.dump(dic)
    else:
        data_str = yaml.dump(dic, default_flow_style=False)
    with open(path, 'w') as f:
        f.write(data_str)
##ondef