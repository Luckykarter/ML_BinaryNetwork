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

def manual_stop(model):
    # The key combination to check
    COMBINATIONS = [
        {keyboard.Key.ctrl, keyboard.KeyCode(char='s')},
        {keyboard.Key.ctrl, keyboard.KeyCode(char='S')}
    ]

    # The currently active modifiers
    current = set()

    def execute():
        listener.stop()
        print("\nTraining stop requested")
        model.stop_training = True

    def on_press(key):
        if any([key in COMBO for COMBO in COMBINATIONS]):
            current.add(key)
            if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
                execute()

    def on_release(key):
        if any([key in COMBO for COMBO in COMBINATIONS]):
            current.remove(key)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        print('To stop training manually - press Ctrl+S')
        listener.join()
