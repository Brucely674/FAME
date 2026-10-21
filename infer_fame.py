import sys
from fame.cli import main
if __name__ == '__main__':
    main(['infer', *sys.argv[1:]])
