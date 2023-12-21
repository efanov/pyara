"""
Module for audio classification
"""
from model import model_eval
from audio_prepare import prediction, prepare_signal
from config import CFG


def predict_audio(file_path
                  ,print_probability = False
                  ,pitch_shift = 0
                  ,width = CFG.width
                  ,sample_rate = CFG.SAMPLE_RATE):
    """
     Функция для предсказания аудио (синтезированного / подлинного).

     Параметры:
         file_path (str): Путь к файлу.

     Возвращает:
         int: Предсказание аудио:
             0: если аудио подлинное
             1: если аудио синтезированное
     """

    model = model_eval()
    signal = prepare_signal(file_path,  pitch_shift, width, sample_rate)

    prediction_of_model, probability = prediction(model, signal)
    if print_probability:
        return f'Answer: {prediction_of_model}; Probability: {abs(probability-prediction_of_model)}'

    #Если print_probability = False
    return prediction_of_model


if __name__ == '__main__':
    print(predict_audio("tests/test_audio/real_0.wav", print_probability=True, pitch_shift=10))
