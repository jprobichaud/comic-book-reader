import base64
import io
import numpy
import os
from PIL import Image
from numpy import asarray
import sys
from doctr.io import DocumentFile
from doctr.models import ocr_predictor, from_hub
import cv2
import torch
import torchaudio
import os
import argparse

class SileroTtsWrapper:

    def __init__(self, lang="fr", model_id="v3_fr", speaker=None, sampling_rate=48000):

        torch.hub.download_url_to_file('https://raw.githubusercontent.com/snakers4/silero-models/master/models.yml',
                                       'latest_silero_models.yml',
                                       progress=False)
        self.model, example = torch.hub.load(repo_or_dir='snakers4/silero-models',
                      model='silero_tts',
                      language=lang,
                      speaker=model_id)

        print(f"we have the following speakers : {self.model.speakers}")
        self.speaker_id = self.model.speakers[0] if speaker is None else speaker
        self.sample_rate=sampling_rate
        self.put_accent = True
        self.put_yo = True

    def get_audio(self, text, ofn=None):
        if len(text) == 0:
            print("no text to convert to voice...")
            return torch.zeros(1), self.sample_rate
        try :
            audio = self.model.apply_tts(text=text,
                              speaker=self.speaker_id,
                              sample_rate=self.sample_rate,
                              put_accent=self.put_accent,
                              put_yo=self.put_yo)
            audio = audio.unsqueeze(0)
            if ofn is not None:
                print(f"saving wav to {ofn}")
                torchaudio.save( ofn, audio, self.sample_rate, encoding="PCM_S", bits_per_sample=16)

            return audio, self.sample_rate
        except Exception as e:
            print("error occured while tranforming text to speech: {e}")
            return torch.zeros(1), self.sample_rate


from comic_book_reader import parseComicSpeechBubbles, segmentPage, findSpeechBubbles

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def center_image(pil_img, new_width=2000, new_height=2000, color = (255,255,255)):
    width, height = pil_img.size
    #new_width = width + right + left
    left = (new_width - width) // 2
    top = (new_height - height) // 2
    #new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result



def get_text_from_result(result, show_confidences=True):
    generated_text = result.export() #as json
    words = []
    for i in generated_text['pages']:
        print(f"page {i['page_idx']} has {len(i['blocks'])}")
        for block in i['blocks']:
            print("block")
            for l in block['lines']:
                for w in l['words']:
                    if show_confidences:
                        words.append(f"{w['value']}@{w['confidence']:.2f}")
                    else:
                        words.append(w['value'])

    return " ".join(words)




class BubbleExtractor:

    def __init__(self, tts_engine):
        self.tts_engine = tts_engine
        #self.bubble_engine = bubble_engine

        model_name = "Felix92/doctr-torch-crnn-mobilenet-v3-large-french"
        det_model_arch = 'db_mobilenet_v3_large'
        self.reco_model = from_hub(model_name)
        self.predictor = ocr_predictor(det_arch=det_model_arch, reco_arch=self.reco_model, pretrained=True)

    def process_file(self, file_filename, output_folder, output_text=True, output_contour=True):
        basename = os.path.basename(file_filename)
        parts = basename.split(".")
        base_motif = ".".join(parts[:-1])

        os.makedirs(output_folder, exist_ok=True)

        # begin processing
        image_data = Image.open(file_filename)
        npimg = asarray(image_data)

        if output_contour:
            out_edit_jpg = os.path.join(output_folder, f"{base_motif}.contours.jpg")
            contours = findSpeechBubbles(npimg)
            cv2.drawContours(npimg, contours, -1, (0, 255, 0), 3)
            _, buffer = cv2.imencode('.jpg', npimg)
            print(f"writing {out_edit_jpg}")
            with open(out_edit_jpg, "wb") as wp:
                wp.write(buffer)

        bubbles = segmentPage(npimg)
        print(f"we have {len(bubbles)} bubbles identified")

        for i in range(len(bubbles)):
            ofn = os.path.join(output_folder, f"{base_motif}.bulle_{i:03d}.jpg")
            print(f"saving bubble {i:3d} to {ofn}")
            cv2.imwrite(ofn, bubbles[i])
            bubble_data = Image.open(ofn)
            bubble_data = center_image(bubble_data)
            bubble_npimg = asarray(bubble_data)

            bubble_npimg =  numpy.expand_dims(bubble_npimg, axis=0)
            results = self.predictor(bubble_npimg)

            generated_text = get_text_from_result(results, False)
            print(f"text: {generated_text}")

            wav_path = os.path.join(output_folder, f"{base_motif}.bulle_{i:03d}.wav")
            tts_engine.get_audio(generated_text, wav_path)

            if output_text:
                ofn_txt = os.path.join(output_folder, f"{base_motif}.bulle_{i:03d}.txt")
                with open(ofn_txt, "w") as fout:
                    fout.write(str(generated_text))





if __name__ == '__main__':
    #croppedImageList = segmentPage(img)
    #pageText = parseComicSpeechBubbles(croppedImageList)


    output_dir = sys.argv[1]
    file_filename = sys.argv[2]

    if os.path.isfile(output_dir):
        print("le premier argument doit être un répertoire")
        sys.exit(1)

    tts_engine = SileroTtsWrapper()
    bubble_extractor = BubbleExtractor(tts_engine)

    bubble_extractor.process_file(file_filename, output_dir)

