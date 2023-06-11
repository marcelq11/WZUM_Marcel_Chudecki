# Opis projektu WZUM
## Przetwarzanie danych
Przetwarzanie danych polega na usunięciu niepotrzebnych kolumn(tj. _handedness.label_, _handedness.score_, oraz pierwszej kolumny z indeksami).
Następnie mapuję wartości _latters_ na wartości liczbowe, aby móc je wykorzystać w dalszej części projektu.
Kolejnym krokiem jest usunięcie 0.3% najbardziej odstających wartości z każdej kolumny, aby
pozbyć się potencjalnie błędnych danych uczących. Następnie dane dzielonę są na zbiory X oraz y.
W przypadku danych treningowych następuje tylko usunięcię niepotrzebnych kolumn.
## Model
ZAimplementowanym modelem jest _svm.SVC(kernel="linear", C=170)_. Parametr _kernel_ wybrany został 
ze względu na ilość danych treningowych oraz zaobserwowane najwyższe wyniki. Natomiast parametr C
został dobrany metodą prób i błędów. 
## Dane treningowe
Dane treningowe zostały zebrane prze całą grupę. Ilość danych wykorzystana do treningu modelu jest równa 
5377 próbek.
