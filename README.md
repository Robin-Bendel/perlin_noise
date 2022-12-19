# Perlin Noise
Dieses Projekt wurde im Rahmen der Vorlesung Parallel Computing erstellt um die Laufzeit der Berechnung von Perlin Noise auf der CPU und der GPU zu vergleichen.
## Verhalten
Das Programm generiert eine Textur mit Perlin Noise. Dies wird jeweils einmal auf der CPU und einmal auf der GPU wiederholt. Dabei wird die Laufzeit gemessen und auf der Kommandozeile ausgegeben.
## Build
Das Programm kann mit dem Befehl `nvcc perlin_noise.cu -o perlin_noise` gebaut werden.
## Ausführen
Beim Start des Programms wird die Größe der zu berechnenden Textur angegeben
 `./perlin_noise 1024`
 
## Umgebung

- Getestet unter Ubuntu 20.04 
- CPU: AMD® Ryzen 7 4800h
- GPU: NVIDIA GeForce GTX 1650 Ti
