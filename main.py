import Attitude_1
import Helpers

att_model = Attitude_1.Model(Helpers.Configuration(epochs=30, batch_size=10), 100, 100)
# att_model.train('/Users/Eric/ML_data/Attitude_1/data_roll/train_data',
#                 '/Users/Eric/ML_data/Attitude_1/data_roll/validation_data',
#                 '/Users/Eric/ML_data/Attitude_1/data_roll/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/Attitude_1/data_roll/prediction_data')
att_model.train('./data_roll/train_data',
                './data_roll/validation_data',
                './data_roll/test_data')
predictions = att_model.predict('./data_roll/prediction_data')
print(predictions)
