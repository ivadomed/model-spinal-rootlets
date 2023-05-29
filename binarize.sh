#!/bin/bash

# Boucle pour les fichiers de 1 à 20
for ((i=1; i<=20; i++)); do
  # Générer le nom de fichier avec deux chiffres
  file_number=$(printf "%03d" $i)
  input_file="mri_${file_number}.nii.gz"
  output_file="mri_${file_number}.nii.gz"

  # Exécuter la commande sct_maths avec les fichiers d'entrée et de sortie
  sct_maths -i "$input_file" -o "$output_file" -bin 1

  echo "Traitement terminé pour ${input_file}"
done
