import Attitude_1
import Helpers

att_model = Attitude_1.Model(Helpers.Configuration(epochs=30, batch_size=10), 100, 100)
# att_model.train('/Users/Eric/ML_data/Attitude_1/data/train_data',
#                 '/Users/Eric/ML_data/Attitude_1/data/validation_data',
#                 '/Users/Eric/ML_data/Attitude_1/data/test_data')
# predictions = att_model.predict('/Users/Eric/ML_data/Attitude_1/data/prediction_data')
att_model.train('./data/train_data',
                './data/validation_data',
                './data/test_data')
predictions = att_model.predict('./data/prediction_data')
print(predictions)
