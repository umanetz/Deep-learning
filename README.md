## Проект по DL и LSML 
*Freesound Audio Tagging 2019*

- Для начала воспроизведем безлайн 

######  epochs=10, bs=32, lr=3e-4

<img src="./img/lrap.png" alt="drawing" width="350"/> <img src="./img/loss.png" alt="drawing" width="350"/>

Очевидно, что модель недообучилась. 

При увеличении эпох до 40 loss улетает

<img src="./img/loss_lrap_base.png" alt="drawing" width="500"/>

Посмотрим на распределение lelrap по классам:

<img src="./img/per_class_lwlrap_base.png" alt="drawing" width="500"/>

Из графика видно, что плохо предсказываются классы 

Далее будем сохранять лучшую модель.
- В следущем эксперементе увеличили число эпох, сделали cycle lr и применили алгоритм mixup 

######  epochs=96, bs=64, lr=cycle

<img src="./img/lr_scheduler.png" alt="drawing" width="500"/>


<img src="./img/loss_lrap_mixup.png" alt="drawing" width="500"/> 

На валидации качество подросло. Это видно на картинке ниже:

<img src="./img/mixup_vs_base.png" alt="drawing" width="500"/>

Также подросло значение per_class_lwlrap для плохо предсказываемых классов у бейзлайна. 


<img src="./img/per_class_lwlrap_mixup.png" alt="drawing" width="500"/>







