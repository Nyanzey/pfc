import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        self.count = 0
        
        now = datetime.now().strftime("%c").replace(' ', '-').replace(':', '-')
        self.log_file = f'{self.log_dir}/log-{now}.txt'


    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f'<{self.count}> [{datetime.now().strftime("%c")}] :' + str(message) + '\n')
            self.count += 1
            