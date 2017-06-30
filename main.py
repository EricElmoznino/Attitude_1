import Attitude_1
import Helpers

att_model = Attitude_1.Model(Helpers.Configuration(epochs=20), 100, 100)
att_model.train('/Users/Eric/ML_data/train_data',
                '/Users/Eric/ML_data/validation_data',
                '/Users/Eric/ML_data/test_data')
predictions = att_model.predict('/Users/Eric/ML_data/prediction_data')
print(predictions)
