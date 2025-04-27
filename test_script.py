from packages.test import test_amass
from packages.utils import argument_parser

test_amass.run(argument_parser.get_train_arguments())