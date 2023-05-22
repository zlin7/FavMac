
import neptune.new as neptune
import _settings


class FakeNeptuneItem():
    def __init__(self) -> None:
        pass
    def log(self, *args, **kwargs):
        pass

class FakeNeptuneSession():
    def __init__(self) -> None:
        self.fakeitem = FakeNeptuneItem()
    def __getitem__(self, key):
        return self.fakeitem
    def __setitem__(self, key, value):
        print(key, value)
        pass
    def stop(self):
        pass

def get_neptune_logger(key, debug=False, continue_from_key=None, tag=[]):
    if debug: return FakeNeptuneSession()
    if continue_from_key is not None:
        project = neptune.get_project(_settings.NEPTUNE_PROJECT, api_token=_settings.NEPTUNE_API_TOKEN)
        _df = project.fetch_runs_table(tag=tag).to_pandas()
        _df = _df[_df['sys/name'].map(lambda s: s.split()[-1] == continue_from_key)]
        # TODO: Note that we cannot continue from the same key anymore..
        if len(_df) == 1:
            _continue_run = _df.iloc[0]['sys/id']
            neptune_ses = neptune.init(project=_settings.NEPTUNE_PROJECT, api_token=_settings.NEPTUNE_API_TOKEN, run=_continue_run)
            neptune_ses['sys/name'] = neptune_ses['sys/name'] + " " + key
        else:
            neptune_ses = neptune.init(project=_settings.NEPTUNE_PROJECT,
                                            api_token=_settings.NEPTUNE_API_TOKEN,
                                            name=key, tags=tag)
            neptune_ses['sys/name'] = f"{continue_from_key} {key}"
    else:
        neptune_ses = neptune.init(project=_settings.NEPTUNE_PROJECT, api_token=_settings.NEPTUNE_API_TOKEN,
                                            name=key, tags=tag)
    return neptune_ses



class FullLogger():
    def __init__(self, logger=None, neptune_ses=None, tbwriter=None) -> None:
        self.logger = logger
        if neptune_ses is None: neptune_ses = FakeNeptuneSession()
        self.neptune_ses = neptune_ses
        self.tbwriter = tbwriter

    def log_scalar(self, key, value, step, log_msg=False):
        if self.tbwriter is not None:
            self.tbwriter.add_scalar(key, value, step)
        self.neptune_ses[key].log(value, step)
        if log_msg:
            self.logger.info("{} = {} at step {}".format(key, value, step))

    def __getitem__(self, key):
        return self.neptune_ses[key]

    def __setitem__(self, key, value):
        self.neptune_ses[key] = value

    def stop(self):
        self.neptune_ses.stop()

    def info(self, msg):
        self.logger.info(msg)
        self.neptune_ses['logs'].log(msg)

    