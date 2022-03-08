
import logging
from watchdog.events import LoggingEventHandler, FileSystemEventHandler

class AnnoEventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def __init__(self, logger=None):
        super().__init__()

        self.logger = logger or logging.root

    def on_moved(self, event):
        super().on_moved(event)

        what = 'directory' if event.is_directory else 'file'
        self.logger.info("Moved %s: from %s to %s", what, event.src_path,
                         event.dest_path)

    def on_created(self, event):
        super().on_created(event)

        what = 'directory' if event.is_directory else 'file'
        self.logger.info("Created %s: %s", what, event.src_path)

    def on_deleted(self, event):
        super().on_deleted(event)

        what = 'directory' if event.is_directory else 'file'
        self.logger.info("Deleted %s: %s", what, event.src_path)

    def on_modified(self, event):
        super().on_modified(event)
        what = 'directory' if event.is_directory else 'file'
        if what is 'file':
            self.logger.info("Annotator detected modified %s: %s", what, event.src_path)
