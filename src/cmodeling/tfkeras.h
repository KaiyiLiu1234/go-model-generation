#ifndef _TFKERAS_H
#define _TFKERAS_H

float LoadExistingTFKerasModelandPredict(char tfkeras_model_filepath[], char **features_float, int float_feature_len, char **label, int label_len, char **features_str, int str_feature_len, float float_features_predict[], char **str_features_predict);

#endif  // TFKERAS_H
