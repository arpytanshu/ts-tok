import json
import sys

class Config:
    def __init__(self, **kwargs):
        path = kwargs.get('path')
        config = kwargs.get('config')
        if path is not None:
            self.load_json(path)
        elif config is not None:
            self.load_state_dict(config)
        else:
            raise Exception("At least one of 'config' or 'path' must be provided.")
    
    def state_dict(self): # get the config values as a dictionary.
        dump_dict = {}
        for k in self.__dict__.keys():
            v = self.__dict__[k]
            if type(v) == type(self):
                dump_dict[k] = v.__dict__
            else:
                dump_dict[k] = v
        return dump_dict

    def save_json(self, path): # save a Config object into a json file.
        dump_dict = self.state_dict()
        with open(path, 'w') as f:
            json.dump(dump_dict, f)
    
    def load_json(self, path): # load a json file into Config object.
        with open(path, 'r') as f:
            load_dict = json.load(f)
        self.load_state_dict(load_dict)
    
    def load_state_dict(self, d): # load a dictionary into Config object.
        for k in d.keys():
            v = d[k]
            if type(v) == dict:
                self.__dict__[k] = Config(config=v)
            else:
                self.__dict__[k] = v

    def __repr__(self):
        s = []
        for k in self.__dict__.keys():
            v = self.__dict__[k]
            if type(v) == type(self): # this is top level config.
                s.append(k)
                s.append(self.__dict__[k].__repr__())
            else: # this is last level config.
                s.append('\t' + k +':\t'+ str(v))
        return '\n'.join(s)
    


def progress_bar(current, total, bar_length=50, text="Progress"):
    percent = float(current) / total
    abs = f"{{{current} / {total}}}"
    arrow = '|' * int(round(percent * bar_length))
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r{0}: [{1}] {2}% {3}".format(text, arrow + spaces, int(round(percent * 100)), abs))
    sys.stdout.flush()
