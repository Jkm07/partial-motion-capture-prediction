from packages.train import train_post_amass
from packages.utils import argument_parser

train_post_amass.run(argument_parser.get_train_arguments())
