import os
import pickle
import inflection


class HealthInsurance(object):
    def __init__(self):
        self.home_path = os.getcwd()

        self.annual_premium_scaler = pickle.load(open(os.path.join(
            self.home_path,
            '/src/features',
            'annual_premium_scaler.pkl'), 'rb'))

        self.age_scaler = pickle.load(open(os.path.join(
                self.home_path,
                '/src/features',
                'age_scaler.pkl'), 'rb'))

        self.vintage_scaler = pickle.load(open(os.path.join(
            self.home_path,
            '/src/features',
            'vintage_scaler.pkl'), 'rb'))

        self.target_encode_gender_scaler = pickle.load(open(os.path.join(
            self.home_path,
            '/src/features',
            'target_encode_gender_scaler.pkl'), 'rb'))

        self.target_encode_region_code_scaler = pickle.load(open(os.path.join(
            self.home_path,
            '/src/features',
            'target_encode_region_code_scaler.pkl'), 'rb'))

        self.fe_policy_sales_channel_scaler = pickle.load(open(os.path.join(
            self.home_path,
            '/src/features', 'fe_policy_sales_channel_scaler.pkl'), 'rb'))

    def rename_columns(self, data):
        new_columns = {col: inflection.underscore(col) for col in data.columns}
        return data.rename(columns=new_columns)

    def feature_engineering(self, data):
        # vehicle age
        # df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 
        # 'over_2_years' if x == '> 2 Years' else 'between_1_2_year' if x ==
        # '1-2 Year' else 'below_1_year')

        # vehicle damage
        data['vehicle_damage'] = (data['vehicle_damage']
                                  .apply(lambda x: 1 if x == 'Yes' else 0))

        return data

    def data_preparation(self, data):
        # annual premium - StandardScaler
        data['annual_premium'] = (self.annual_premium_scaler
                                  .transform(data[['annual_premium']].values))

        # Age - MinMaxScaler
        data['age'] = self.age_scaler.transform(data[['age']].values)

        # vintage
        data['vintage'] = (self.vintage_scaler.transform(data[['vintage']]
                                                         .values))

        # gender - Target Encoding
        # data.loc[:, 'gender'] = (data['gender']
        #                          .map(self.target_encode_gender_scaler))

        # region_code - Target Encoding
        data.loc[:, 'region_code'] = (data['region_code'].map(
            self.target_encode_region_code_scaler))

        # # vehicle_age - One Hot Encoding
        # data = (pd.get_dummies(data, prefix='vehicle_age',
        #                        columns=['vehicle_age']))

        # policy_sales_channel - Target Encoding / Frequency Encoding (!)
        data.loc[:, 'policy_sales_channel'] = (
            data['policy_sales_channel']
            .map(self.fe_policy_sales_channel_scaler))

        # Feature selection
        cols_selected = ['vintage', 'annual_premium', 'age', 'region_code',
                         'vehicle_damage', 'policy_sales_channel',
                         'previously_insured']

        return data[cols_selected]

    def get_prediction(self, model, original_data, test_data):
        # model prediction
        pred = model.predict_proba(test_data)

        # join prediction into original data
        original_data['score'] = pred[:, 1].tolist()

        return original_data.to_json(orient='records', date_format='iso')
