# CarPricePredictor-ML: сервис для предсказания стоимости машины

## Скриншоты работы FastAPI по пути `/predict_item` на примере одного объекта из тестовой выборки

Файл с примером: `single_item.json`

- Post-запрос и request для одного объекта:
  ![image](https://github.com/user-attachments/assets/c6c58439-55b4-4c35-817b-c937920a1936)
  ![image](https://github.com/user-attachments/assets/17658721-c7cc-4214-aa43-3178b43c5c22)

## Скриншоты работы FastAPI по пути `/predict_items` на примере нескольких объектов из тестовой выборки

Файл с примером: `multiple_items.json`

- Post-запрос и request для нескольких объектов:
  ![image](https://github.com/user-attachments/assets/3b350890-0372-44e2-999f-0e4b02c5ba68)
  ![image](https://github.com/user-attachments/assets/dcb42de6-af2a-4ce2-ae4c-87aa8a907b03)

## Результат post-запрос по пути `/predict_items` в виде csv-файла

После обработки файл с признаками тестовых объектов дополнен новым столбцом с предсказаниями.

- Пример: 
  ![image](https://github.com/user-attachments/assets/0b87d116-1c3b-42dd-8915-ec94745d4373)
