from tensorflow.keras.callbacks import Callback

class stopTraining(Callback):
    def __init__(self, accuracy):
        super().__init__()
        self.accuracy = accuracy
    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('acc')
        if accuracy is None:
            accuracy = logs.get('accuracy')
        if accuracy > self.accuracy:
            print('\nReached accuracy {}% - stopped training'.
                  format(accuracy*100))
            self.model.stop_training = True

class stopOnOptimalAccuracy(Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        if val_loss >= loss:
            print('\nReached optimal validation accuracy {}% - stopped training'.
                  format(logs.get('val_accuracy')*100))
            self.model.stop_training = True

