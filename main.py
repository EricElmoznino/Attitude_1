import Attitude_1
import Helpers

att_model = Attitude_1.Model(Helpers.Configuration(epochs=20), 100, 100)
# att_model.train('/Users/Eric/ML_data/Attitude_1/train_data',
#                 '/Users/Eric/ML_data/Attitude_1/validation_data',
#                 '/Users/Eric/ML_data/Attitude_1/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/prediction_data')
att_model.train('./Attitude_1_data/train_data',
                './Attitude_1_data/validation_data',
                './Attitude_1_data/test_data')
predictions = att_model.predict('./Attitude_1_data/prediction_data')
print(predictions)
