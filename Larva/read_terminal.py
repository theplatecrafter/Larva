import sys

class TerminalOutputWatcher:
    def __init__(self, file_path):
        self.file_path = file_path
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.output_file = open(self.file_path, 'w')
        self.is_watching = False

    def start_watching(self):
        if not self.is_watching:
            sys.stdout = self.output_file
            sys.stderr = self.output_file
            self.is_watching = True

    def stop_watching(self):
        if self.is_watching:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.output_file.close()
            self.is_watching = False

# Usage example
watcher = TerminalOutputWatcher("terminal_output.txt")

def startreadingterminal():
    watcher.start_watching()

def stopreadingterminal():
    watcher.stop_watching()