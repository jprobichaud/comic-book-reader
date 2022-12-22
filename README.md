# Comic Book Page Reader

## Summary
Application python pour extraire les bulles et le texte de celles-ci avant de générer des fichiers audios correspondants.


## Installation

C'est un peu primitif pour l'instant.

1. Installer conda ()
1. Créer l'environment : `conda env create -f conda_environment.yml`
1. Activer l'environment conda : `conda activate ComicReaderFr`

## Example:

```
python from-folder.py repertoire_de_sortie  fichier_image.jpg
```

Le script va analyzer le fichier `fichier_image.jpg` et mettre les fichiers audios, le fichier html, les différentes images et le texte extract (un fichier par bulle détectée) dans le répertoire, avec comme base de nom `fichier_image` 

Il y a quelques fichiers dans le répertoire `examples` 



## Technologies
Python, Pytorch, 

## Libraries
doctr.io for OCR (Optical Character Recognition), OpenCV
snakers4/silero-models for TTS generation


## Developer Notes


## References

Many thanks to: https://github.com/damishshah/comic-book-reader !

Dubray, David & Laubrock, Jochen. (2019). Deep CNN-based Speech Balloon Detection and Segmentation for Comic Books. https://arxiv.org/abs/1902.08137.

>> I wanted to give a shoutout to the research team from Cornell for their research in this area. I reached out to them when I was considering a Nueral Net approach to this problem and they helped answer some questions that I had. You can read their excellent research paper at the above link and check out their model here: https://github.com/DRDRD18/balloons
