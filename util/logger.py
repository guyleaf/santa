class SimpleLogger:
    def __init__(self, path):
        self.path = path

    def log(self, iteration, max_iteration, metric_dict, verbose=False):
        message = "[%03d/%03d] " % (iteration, max_iteration)
        for key in metric_dict:
            message += "\t %s:%.3f \t" % (key, metric_dict[key])
        if verbose:
            print(message)
        record = open(self.path, "a")
        record.write("\n" + message + "\n")
        record.close()

    def log_message(self, message, verbose=True):
        record = open(self.path, "a")
        record.write("\n" + message + "\n")
        if verbose:
            print(message)
        record.close()
