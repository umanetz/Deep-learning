## Проект по DL и LSML 
*Freesound Audio Tagging 2019*

Для начала воспроизведем безлайн 

######  epochs=10, bs=32, lr=3e-4

<img src="./img/lrap.png" alt="drawing" width="350"/> <img src="./img/loss.png" alt="drawing" width="350"/>

Очевидно, что модель недообучилась. 

Следущем эксперементе увеличили число эпох, зделали cycle lr и применили алгоритм mixup 

######  epochs=96, bs=64, lr=cycle

<img src="./img/cycle_lr.png" alt="drawing" width="350"/>
<img src="./img/lrap_mixup.png" alt="drawing" width="350"/> <img src="./img/loss_mixup.png" alt="drawing" width="350"/>




