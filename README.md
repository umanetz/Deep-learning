## Проект по DL и LSML 
*Freesound Audio Tagging 2019*

- Для начала воспроизведем безлайн 

######  epochs=10, bs=32, lr=3e-4

<img src="./img/lrap.png" alt="drawing" width="350"/> <img src="./img/loss.png" alt="drawing" width="350"/>

Очевидно, что модель недообучилась. 

При увеличении эпох до 40 loss улетает



<img src="./img/loss_lrap_base.png" alt="drawing" width="650"/>

Далее будем сохранять лучшую модель.
- В следущем эксперементе увеличили число эпох, сделали cycle lr и применили алгоритм mixup 

######  epochs=96, bs=64, lr=cycle

<img src="./img/lr_scheduler.png" alt="drawing" width="350"/>


<img src="./img/loss_lrap_mixup.png" alt="drawing" width="350"/> 

Примерно с 30 эпохи начинается переобучение. Также видна зависимость от циклов lr.




