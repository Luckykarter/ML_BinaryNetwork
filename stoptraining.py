from tensorflow.keras.callbacks import Callback
from pynput import keyboard


class StopTraining(Callback):
    def __init__(self, accuracy):
        super().__init__()
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('acc')
        if accuracy is None:
            accuracy = logs.get('accuracy')
        if accuracy > self.accuracy:
            print('\nReached accuracy {}% - stopped training'.
                  format(accuracy * 100))
            self.model.stop_training = True

# class for manual stop via hotkey (Ctrl+S)
class ManualStop:
    def __init__(self, model):
        self.model = model
        self.listener = None

        # The currently active modifiers
        self.current = set()

        # The key combination to check
        self.COMBINATIONS = [
            {keyboard.Key.ctrl, keyboard.KeyCode(char='s')},
            {keyboard.Key.ctrl, keyboard.KeyCode(char='S')}
        ]

    def start_listener(self):
        with keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as self.listener:
            print('To stop training manually - press Ctrl+S')
            self.listener.join()

    def stop_listener(self):
        self.listener.stop()

    def execute(self):
        self.listener.stop()
        print("\nTraining stop requested")
        self.model.stop_training = True

    def on_press(self, key):
        if any([key in COMBO for COMBO in self.COMBINATIONS]):
            self.current.add(key)
            if any(all(k in self.current for k in COMBO) for COMBO in self.COMBINATIONS):
                self.execute()

    def on_release(self, key):
        if any([key in COMBO for COMBO in self.COMBINATIONS]):
            self.current.remove(key)
