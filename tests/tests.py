import unittest

import torch
import torchaudio
import numpy as np

from pyara.main import predict_audio
from pyara.Model.model import ResNetBlock, MFCCModel, model_eval
from pyara.config import CFG
from pyara.audio_prepare import cut_if_necessary, right_pad_if_necessary, prepare_signal, prediction
from pyara.audio_prepare import MFCC_spectrogram, prediction_multiple, prepare_signals

''' Для правильного запуска тестов необходимо сначала установить библиотеку pyara: 
        pip install pyara 
    и перейти в текущую директорию для правильных путей к аудиозаписям'''

# Model test (ResNet)
class TestResNetBlock(unittest.TestCase):
    def setUp(self):
        self.block = ResNetBlock(16, 16)

    def test_forward(self):
        """Тест ResNet"""
        batch_size = 16
        in_depth = 16
        signal = torch.randn(batch_size, in_depth, 4, 4)
        output = self.block(signal)
        self.assertEqual(output.shape, (batch_size, in_depth, 2, 2))


# Model test (MFCCModel)
class TestMFCCModel(unittest.TestCase):
    def setUp(self):
        self.model = MFCCModel()

    def test_forward(self):
        """Тест MFCCModel"""
        batch_size = 8
        in_channels = 1
        T = 100
        signal = torch.randn(batch_size, in_channels, T)
        output = self.model(signal)
        self.assertEqual(output.shape, (batch_size, 2))


# Model evaluating test
class TestModelEval(unittest.TestCase):
    def test_model_eval(self):
        """Тест правильности загрузки модели"""
        model = model_eval()
        self.assertIsInstance(model, MFCCModel)


# Different audio formats test
class TestAudioFormats(unittest.TestCase):
    def test_mp3(self):
        """Тест работы с форматом mp3"""
        self.assertIn(predict_audio('test_audio/1.mp3'), (0, 1))

    def test_wav(self):
        """Тест работы с форматом wav"""
        self.assertIn(predict_audio('test_audio/1.wav'), (0, 1))

    def test_flac(self):
        """Тест работы с форматом flac"""
        self.assertIn(predict_audio('test_audio/1.flac'), (0, 1))

    def test_aiff(self):
        """Тест работы с форматом aiff"""

        self.assertIn(predict_audio('test_audio/1.aiff'), (0, 1))

    def test_ogg(self):
        """Тест работы с форматом ogg"""
        self.assertIn(predict_audio('test_audio/1.ogg'), (0, 1))

    def test_mp3_wav(self):
        """Тест работы с неправильным расширением файла"""
        self.assertIn(predict_audio('test_audio/2.wav'), (0, 1))


# Correct prediction of real and fake voices test
class TestPredictAudio(unittest.TestCase):

    def test_real_voice(self):
        """Тест предсказания реального аудио"""

        file_path = "test_audio/real_0.wav"
        result = predict_audio(file_path)

        self.assertEqual(result, 0, "Expected real voice prediction")

    def test_synthesized_voice(self):
        """Тест предсказания синтезированного аудио"""

        file_path = "test_audio/Alg_1_0.wav"
        result = predict_audio(file_path)

        self.assertEqual(result, 1, "Expected synthesized voice prediction")


# Signal processing function Test
class TestFunctions(unittest.TestCase):
    def test_cut_if_necessary(self):
        """Тест функции обрезающей сигнал до нужной ширины, если ширина сигнала
         больше чем CFG.width или другая заданная пользователем ширина сигнала"""
        # Test case when signal.shape[2] > CFG.width
        signal = torch.randn(2, 2, 400)
        expected_output = signal[:, :, 0:CFG.width]
        self.assertTrue(np.array_equal(cut_if_necessary(signal).numpy(), expected_output.numpy()))

        # Test case when signal.shape[2] <= CFG.width
        signal = torch.randn(2, 2, 50)
        self.assertTrue(np.array_equal(cut_if_necessary(signal).numpy(), signal.numpy()))

    def test_right_pad_if_necessary(self):
        """Тест функции дополняющей сигнал до нужной ширины, если ширина сигнала
         меньше чем CFG.width или другая заданная пользователем ширина сигнала"""
        # Тест когда length_signal < CFG.width
        signal = torch.randn(1, 1, 50)
        expected_output = torch.nn.functional.pad(signal, (0, CFG.width - 50))
        self.assertTrue(np.array_equal(right_pad_if_necessary(signal).numpy(), expected_output.numpy()))

        # Тест когда length_signal >= CFG.width
        signal = torch.randn(1, 1, 300)
        self.assertTrue(np.array_equal(right_pad_if_necessary(signal).numpy(), signal.numpy()))

    def test_prepare_signal(self):
        """Тест полного цикла предобработки аудио"""
        voice_path = "test_audio/real_0.wav"
        signal, sample_rate = torchaudio.load(voice_path)
        expected_output = signal.mean(dim=0)
        expected_output = expected_output.unsqueeze(dim=0)
        expected_output = MFCC_spectrogram(expected_output)
        expected_output = cut_if_necessary(expected_output)
        expected_output = right_pad_if_necessary(expected_output)
        expected_output = expected_output.repeat(3, 1, 1)
        expected_output = expected_output.unsqueeze(dim=0)
        expected_output = expected_output.to(CFG.device)

        self.assertTrue(np.array_equal(prepare_signal(voice_path).numpy(), expected_output.numpy()))

    def test_prediction_multiple(self):
        """Тест предсказания нескольких аудио"""
        model = model_eval()
        signal = prepare_signals(['test_audio/real_0.wav', 'test_audio/real_0.wav'], 0, 300, 16000)
        prediction_of_model, probability = prediction_multiple(model, signal)
        print(prediction_of_model)
        self.assertIn(prediction_of_model[0], (0, 1))
        self.assertIn(prediction_of_model[1], (0, 1))

    def test_prediction(self):
        """Тест предсказания одного аудио"""
        model = model_eval()
        signal = prepare_signal('test_audio/real_0.wav', 0, 300, 16000)
        prediction_of_model, probability = prediction(model, signal)
        print(prediction_of_model)
        self.assertIn(prediction_of_model, (0, 1))
        self.assertTrue(int(probability) >= 0 | int(probability) <= 1)

if __name__ == '__main__':
    unittest.main()
