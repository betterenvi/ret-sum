"""Main."""
import logging
import os
import sys

from common.controller import Controller
from common.util import check_dir
from common.util import get_cur_git_rev
from common.util import save_configs

from .model.combined import ApiDocVAE
from .model.config import ApiDocVAEConfig
from .model.config import ApiMLPConfig
from .model.config import ApiMLPERNNDConfig
from .model.config import DocMLPConfig
from .model.config import DocMLPERNNDConfig
from .model.data_config import RNSQLData
from .model.data_config import SQLData
from .model.mlp import MLPModel
from .model.mlpe_rnnd import MLPERNNDModel


if __name__ == '__main__':
    config = ApiDocVAEConfig(args=sys.argv, prefix='')

    model_dict = {
        'mlp': MLPModel,
        'mlpe_rnnd': MLPERNNDModel,
    }
    api_config_dict = {
        'mlp': ApiMLPConfig,
        'mlpe_rnnd': ApiMLPERNNDConfig,
    }
    doc_config_dict = {
        'mlp': DocMLPConfig,
        'mlpe_rnnd': DocMLPERNNDConfig,
    }
    data_config_dict = {
        'sql': SQLData,
        'rnsql': RNSQLData,
    }
    if config.switch_data_to:
        data_config = data_config_dict[config.switch_data_to](args=sys.argv)
        config.switch_data(data_config)

    api_model = config.api_model.lower()
    doc_model = config.doc_model.lower()

    api_config = api_config_dict[api_model](args=sys.argv, prefix='api')
    doc_config = doc_config_dict[doc_model](args=sys.argv, prefix='doc')

    check_dir(config.model_path)

    # Get current cmd and git rev to make a snapshot.
    cmd = 'python3 -m api_emb.main ' + ' '.join(sys.argv[1:])
    rev = get_cur_git_rev()
    msg = '"cmd": "{}",\n"rev": "{}"'.format(cmd, rev)
    save_configs([config, api_config, doc_config],
                 ['config', 'api_config', 'doc_config'],
                 path=os.path.join(config.model_path, config.config_file),
                 msg=msg)

    config.api_config = api_config
    config.doc_config = doc_config

    controller = Controller(config, ApiDocVAE,
                            api_model=model_dict[api_model],
                            doc_model=model_dict[doc_model])
    logging.info('*' * 80)
    logging.warn('Un-parsed args: {}'
                 .format(str(set(sys.argv[1:]) - set(config.parsed_args))))
    logging.info('*' * 80)

    logging.info('Using {} as api model, {} as doc model.'
                 .format(api_model, doc_model))

    controller.run()
