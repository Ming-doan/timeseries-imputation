# Ghost Time Serise Training Modules

Project of `AIL303m` FPT University Da Nang. For conference paper.

## ⚡Models Aliases

- `lr` : Linear Regression
- `knn` : K-Nearest Neighbor
- `svm` : Support Vector Machine
- `dt` : Decision Tree
- `et` : Extra Tree
- `ada` : AdaBoost
- `bag` : Bagging
- `gb` : Gradient Boosting
- `rf` : Random Forest
- `xgb` : XGBoost
- `vote` : Voting of Machine Learning
- `jeong` : Jeong Stacking
- `rnn` : Recurrent Neural Network
- `lstm` : Long-Short Term Memory
- `cnn1d` : Convolution Network 1D
- `gans` : Generative Adversarial Network

## 📖 Documentation

### Training on Local

- Clone project

```bash
# Clone project
git clone https://github.com/Ming-doan/timeseries-imputation.git timeseries
# Change directory to project
cd timeseries
# Install dependencies
pip install -r requirements.txt
```

- Select model to train. On file `ghost.ipynb`

```py
...
# Add the model alias on 👇 this list
MODELS = get_by_aliases(['rf', 'svm'], rf_n_estimators=100)
# Add model custom parameters .........☝️. Start with prefix of model alias.
```

- Add your own model. Create the new file in `modules/models`. In `your_model.py`.

```py
from ._base import BaseModelWrapper
# Define class of your custom model
class YourModel(BaseModelWrapper):
    # Define the name of your model. This name will display during training or plotting results.
    name = "MyCustomModel"

    def __init__(self, **kwargs):
        self.is_generate = False
        # Define your arguments here
        ...

    def fit(self, generator, x, y):
        # If `is_generate` is False, this method will received x, y as `np.NDArray`.
        # If `is_generate` is True, this method will received generator as `WindowGenerator`.
        # The window generator methods is described below.
        ...

    def predict(self, generator, x):
        # If `is_generate` is False, this method will received x as `np.NDArray`.
        # If `is_generate` is True, this method will received generator as `WindowGenerator`.
        ...

    def forecast(self, x, steps):
        # The forecast function is used to forecast the future values.
        # The x is the last window of data as `np.NDArray`.
        # The steps is how many data must be generated.
        ...

    def summary(self):
        # Show the summary of your model
        ...

    def reset(self):
        # The model will run over more than one time. Using the reset method to reset the parameters of your model for the new dataset.
        ...
```

```py
# Adding to `modules/models/__init__.py` for easy importing
from .your_model import YourModel

# Adding to `modules/alias.py` for creating the alias for your model
def get_by_alias(alias, **kwargs):
    ...
    if alias == 'your_defined_alias':
        # By passing kwargs into your model. You will use model custom paramerter on above.
        return YourModel(**filter_and_format('your_defined_alias', kwargs))
    ...
```

- Using `WindowGenerator`. Window Generator is the dynamic generator for creating TimeSeries dataset with window size and batch size.

```py
generator = WindowGenerator(df, WINDOW_SIZE, BATCH_SIZE)
# Using slice index to get the batch of data has been splited by window.
generator[0] # The output shape is (batch_size, window_size, n_features)
# Using `generate()` to generate full data without using batch.
generator.generate() # The output shape is (data_length, window_size, n_features)
```

```py
# You can use as iterator to loop throught data with batch size
for data_with_batch in generator:
    print(data_with_batch.shape) # Output: (batch_size, window_size, n_features)
```

- Define custom preprocessing

```py
from modules.utils.preprocessing import Plugins
# Create class of custom preprocessing
class YourPreprocessing(Plugins):
    def __init__(self): ...

    # Define your algorithm for do preprocessing
    # `x` is `np.NDArray`
    def flow(self, x): ...

    # Define your algorithm for reverse back to original
    # `x` is `np.NDArray`
    def reverse_flow(self, x): ...
```

```py
# Add your preprocessing into use
preprocessor = Preprocessing()
preprocessor.add_plugin(YourPreprocessing())
preprocessor.flow(dataframe)
```

- Custom Training callback

```py
from modules.utils.callbacks import Callback
# Define your custom callback
class YourCallback(Callback):
    def __init__(self):
        super().__init__()

    # Define algorithm after model predicting results
    # Both `y_true` and `y_pred` are `np.NDArray`
    def after_predict(self, y_true, y_pred): ...

    # Define algorithm after model forecasting results
    # Both `y_true` and `y_fore` are `np.NDArray`
    def after_forecast(self, y_true, y_fore): ...
```

```py
# Using callback
your_callback = YourCallback()
trainer.train(generator, callbacks=[your_callback])
```

### Training on Colab

👉 [Phantom Colab](https://colab.research.google.com/drive/1hokWxs8VnsdT_CMmTas-qHiO9KuARAVx?usp=sharing)

## 🕹️ Contributors

<div style="display: flex;">
<a href="https://www.facebook.com/tony2802.D" style="display:flex;flex-direction:column;align-items:center;margin-right:20px"><img src="https://scontent.fsgn2-7.fna.fbcdn.net/v/t39.30808-6/344377948_775593660613596_1216234811214449721_n.jpg?_nc_cat=109&ccb=1-7&_nc_sid=5f2048&_nc_ohc=oFEidQ80HdEAX8RZz7j&_nc_ht=scontent.fsgn2-7.fna&oh=00_AfAVCgJwlxQs1AYI0t4fYGGvwP3N1rQL06vBJ26aHpvoDw&oe=653DBD8E" style="border-radius: 50%; width:50px"/>Đào Ngọc Huy</a>
<a href="https://www.facebook.com/ming.doan/" style="display:flex;flex-direction:column;align-items:center;margin-right:20px"><img src="https://scontent.fsgn2-3.fna.fbcdn.net/v/t39.30808-6/394525696_1124443801865486_8739919297837929580_n.jpg?_nc_cat=107&ccb=1-7&_nc_sid=5f2048&_nc_ohc=h4ImwNoefIYAX9-DM-K&_nc_ht=scontent.fsgn2-3.fna&oh=00_AfA14s5v0VBvrVYxB_5scKkPcLOVzIZGdV5uGbvfy0V1Kw&oe=653E6599" style="border-radius: 50%; width:50px"/>Đoàn Quang Minh</a>
<a href="https://www.facebook.com/quang.dinh.90813236" style="display:flex;flex-direction:column;align-items:center;margin-right:20px"><img src="https://scontent.fsgn2-4.fna.fbcdn.net/v/t39.30808-6/290879595_1475124582919075_7632900938360005513_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=5f2048&_nc_ohc=RQbiD4jXOvkAX9c3ml4&_nc_ht=scontent.fsgn2-4.fna&oh=00_AfDvCjxb3QqlknIT6a2aKtPskKZIJo4X5zyZZtw1Ip-3RQ&oe=653EA896" style="border-radius: 50%; width:50px"/>Đinh Thiều Quang</a>
<a href="https://www.facebook.com/profile.php?id=100037350121063" style="display:flex;flex-direction:column;align-items:center;margin-right:20px"><img src="https://scontent.fsgn2-4.fna.fbcdn.net/v/t39.30808-6/325673426_907218923604608_154990822737406390_n.jpg?_nc_cat=101&ccb=1-7&_nc_sid=5f2048&_nc_ohc=0Wuogv9WZnEAX-jQ9VW&_nc_ht=scontent.fsgn2-4.fna&oh=00_AfCQeso1galiGrrMq2OBr5EGKSyGYXs283m3d81ZE3In1Q&oe=653DEDE5" style="border-radius: 50%; width:50px"/>
Nguyễn Quân Bảo</a>
</div>
