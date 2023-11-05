{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "P8npiudCeust"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.linear_model import LinearRegression\n",
        "from sklearn.metrics import r2_score\n",
        "import pickle\n",
        "from os.path import isfile\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "pwl59rr6fvyY"
      },
      "outputs": [],
      "source": [
        "my_df=pd.read_csv('FuelConsumption.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 274
        },
        "id": "7lJbIKWXgOm4",
        "outputId": "0088a67a-4cf6-4928-8343-1572d292b8db"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>MODELYEAR</th>\n",
              "      <th>MAKE</th>\n",
              "      <th>MODEL</th>\n",
              "      <th>VEHICLECLASS</th>\n",
              "      <th>ENGINESIZE</th>\n",
              "      <th>CYLINDERS</th>\n",
              "      <th>TRANSMISSION</th>\n",
              "      <th>FUELTYPE</th>\n",
              "      <th>FUELCONSUMPTION_CITY</th>\n",
              "      <th>FUELCONSUMPTION_HWY</th>\n",
              "      <th>FUELCONSUMPTION_COMB</th>\n",
              "      <th>FUELCONSUMPTION_COMB_MPG</th>\n",
              "      <th>CO2EMISSIONS</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2014</td>\n",
              "      <td>ACURA</td>\n",
              "      <td>ILX</td>\n",
              "      <td>COMPACT</td>\n",
              "      <td>2.0</td>\n",
              "      <td>4</td>\n",
              "      <td>AS5</td>\n",
              "      <td>Z</td>\n",
              "      <td>9.9</td>\n",
              "      <td>6.7</td>\n",
              "      <td>8.5</td>\n",
              "      <td>33</td>\n",
              "      <td>196</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2014</td>\n",
              "      <td>ACURA</td>\n",
              "      <td>ILX</td>\n",
              "      <td>COMPACT</td>\n",
              "      <td>2.4</td>\n",
              "      <td>4</td>\n",
              "      <td>M6</td>\n",
              "      <td>Z</td>\n",
              "      <td>11.2</td>\n",
              "      <td>7.7</td>\n",
              "      <td>9.6</td>\n",
              "      <td>29</td>\n",
              "      <td>221</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2014</td>\n",
              "      <td>ACURA</td>\n",
              "      <td>ILX HYBRID</td>\n",
              "      <td>COMPACT</td>\n",
              "      <td>1.5</td>\n",
              "      <td>4</td>\n",
              "      <td>AV7</td>\n",
              "      <td>Z</td>\n",
              "      <td>6.0</td>\n",
              "      <td>5.8</td>\n",
              "      <td>5.9</td>\n",
              "      <td>48</td>\n",
              "      <td>136</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2014</td>\n",
              "      <td>ACURA</td>\n",
              "      <td>MDX 4WD</td>\n",
              "      <td>SUV - SMALL</td>\n",
              "      <td>3.5</td>\n",
              "      <td>6</td>\n",
              "      <td>AS6</td>\n",
              "      <td>Z</td>\n",
              "      <td>12.7</td>\n",
              "      <td>9.1</td>\n",
              "      <td>11.1</td>\n",
              "      <td>25</td>\n",
              "      <td>255</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>2014</td>\n",
              "      <td>ACURA</td>\n",
              "      <td>RDX AWD</td>\n",
              "      <td>SUV - SMALL</td>\n",
              "      <td>3.5</td>\n",
              "      <td>6</td>\n",
              "      <td>AS6</td>\n",
              "      <td>Z</td>\n",
              "      <td>12.1</td>\n",
              "      <td>8.7</td>\n",
              "      <td>10.6</td>\n",
              "      <td>27</td>\n",
              "      <td>244</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   MODELYEAR   MAKE       MODEL VEHICLECLASS  ENGINESIZE  CYLINDERS  \\\n",
              "0       2014  ACURA         ILX      COMPACT         2.0          4   \n",
              "1       2014  ACURA         ILX      COMPACT         2.4          4   \n",
              "2       2014  ACURA  ILX HYBRID      COMPACT         1.5          4   \n",
              "3       2014  ACURA     MDX 4WD  SUV - SMALL         3.5          6   \n",
              "4       2014  ACURA     RDX AWD  SUV - SMALL         3.5          6   \n",
              "\n",
              "  TRANSMISSION FUELTYPE  FUELCONSUMPTION_CITY  FUELCONSUMPTION_HWY  \\\n",
              "0          AS5        Z                   9.9                  6.7   \n",
              "1           M6        Z                  11.2                  7.7   \n",
              "2          AV7        Z                   6.0                  5.8   \n",
              "3          AS6        Z                  12.7                  9.1   \n",
              "4          AS6        Z                  12.1                  8.7   \n",
              "\n",
              "   FUELCONSUMPTION_COMB  FUELCONSUMPTION_COMB_MPG  CO2EMISSIONS  \n",
              "0                   8.5                        33           196  \n",
              "1                   9.6                        29           221  \n",
              "2                   5.9                        48           136  \n",
              "3                  11.1                        25           255  \n",
              "4                  10.6                        27           244  "
            ]
          },
          "execution_count": 3,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "my_df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "id": "nH73gTGogWdK"
      },
      "outputs": [],
      "source": [
        "X=my_df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 203
        },
        "id": "dAZnZcbqgybC",
        "outputId": "a131b18d-45bb-4788-beec-d2bb26e47ef1"
      },
      "outputs": [
        {
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>ENGINESIZE</th>\n",
              "      <th>CYLINDERS</th>\n",
              "      <th>FUELCONSUMPTION_COMB</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>2.0</td>\n",
              "      <td>4</td>\n",
              "      <td>8.5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>2.4</td>\n",
              "      <td>4</td>\n",
              "      <td>9.6</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1.5</td>\n",
              "      <td>4</td>\n",
              "      <td>5.9</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3.5</td>\n",
              "      <td>6</td>\n",
              "      <td>11.1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>3.5</td>\n",
              "      <td>6</td>\n",
              "      <td>10.6</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   ENGINESIZE  CYLINDERS  FUELCONSUMPTION_COMB\n",
              "0         2.0          4                   8.5\n",
              "1         2.4          4                   9.6\n",
              "2         1.5          4                   5.9\n",
              "3         3.5          6                  11.1\n",
              "4         3.5          6                  10.6"
            ]
          },
          "execution_count": 5,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "X.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "L9jkJ35ag9GW"
      },
      "outputs": [],
      "source": [
        "y=my_df['CO2EMISSIONS']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2zd6CCYUhEKm",
        "outputId": "e4ea4bca-0618-41d0-df45-21c6876776e0"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0    196\n",
              "1    221\n",
              "2    136\n",
              "3    255\n",
              "4    244\n",
              "Name: CO2EMISSIONS, dtype: int64"
            ]
          },
          "execution_count": 7,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "y.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "q7D5zJRIhKHK"
      },
      "outputs": [],
      "source": [
        "X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.3,random_state=4)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "Ly4zrgSThnX5"
      },
      "outputs": [],
      "source": [
        "model = LinearRegression()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MrnYAzO0htrU",
        "outputId": "6dfbfe9b-176e-4e41-ec25-10036fa77401"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "execution_count": 15,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "model.fit(X_train,y_train)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "8Xdj9DJ6hy4F"
      },
      "outputs": [],
      "source": [
        "y_pred =model.predict(X_test)\n",
        "accurcy = r2_score(y_test,y_pred)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pSUqwqiAiFkC",
        "outputId": "8d799451-3f62-4514-9d97-e5664b911649"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.8695432015478277"
            ]
          },
          "execution_count": 17,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "accurcy\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 31,
      "metadata": {
        "id": "iPaEn6DHiG1J"
      },
      "outputs": [],
      "source": [
        "file_name ='co2_em.pickle'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "metadata": {
        "id": "r2wrzvxki7nW"
      },
      "outputs": [],
      "source": [
        "top_acc=0"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 32,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-QJPxbPAiXlQ",
        "outputId": "8d1dcfe9-f6d8-4b0a-8e5a-b8f969360524"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "the accurcy is :  0.8490552129718008\n",
            "the accurcy is :  0.8550924013666573\n",
            "the accurcy is :  0.8626928888348232\n",
            "the accurcy is :  0.8593147640957989\n",
            "the accurcy is :  0.8704159909351121\n",
            "the accurcy is :  0.8798316422398744\n",
            "the accurcy is :  0.8542358839114924\n",
            "the accurcy is :  0.8713041329205731\n",
            "the accurcy is :  0.8431271583374723\n",
            "the accurcy is :  0.855860303499768\n",
            "the accurcy is :  0.870324936088811\n",
            "the accurcy is :  0.8553117540294763\n",
            "the accurcy is :  0.848534584354485\n",
            "the accurcy is :  0.8428146682274312\n",
            "the accurcy is :  0.8635330790015864\n",
            "the accurcy is :  0.8595784486364529\n",
            "the accurcy is :  0.8791431777789239\n",
            "the accurcy is :  0.874883481289845\n",
            "the accurcy is :  0.8400218448179654\n",
            "the accurcy is :  0.8493575910584015\n",
            "the accurcy is :  0.8626533567037595\n",
            "the accurcy is :  0.8573579289208403\n",
            "the accurcy is :  0.8861656208522782\n",
            "the accurcy is :  0.8742607093794312\n",
            "the accurcy is :  0.8700878602450913\n",
            "the accurcy is :  0.8538056387167178\n",
            "the accurcy is :  0.8773004351748199\n",
            "the accurcy is :  0.8616643517801927\n",
            "the accurcy is :  0.8749644872358119\n",
            "the accurcy is :  0.8542367981908348\n",
            "the accurcy is :  0.861179691350752\n",
            "the accurcy is :  0.8774624237621721\n",
            "the accurcy is :  0.8663594000980345\n",
            "the accurcy is :  0.8781816281214944\n",
            "the accurcy is :  0.8785099019103678\n",
            "the accurcy is :  0.8474886087179218\n",
            "the accurcy is :  0.8684909146491226\n",
            "the accurcy is :  0.8601951557492009\n",
            "the accurcy is :  0.866524744835238\n",
            "the accurcy is :  0.8630168115571606\n",
            "the accurcy is :  0.863448750040881\n",
            "the accurcy is :  0.8725908807734583\n",
            "the accurcy is :  0.8680733952211733\n",
            "the accurcy is :  0.8452541638918059\n",
            "the accurcy is :  0.8528435698493767\n",
            "the accurcy is :  0.8736547754396861\n",
            "the accurcy is :  0.8509786622762032\n",
            "the accurcy is :  0.8721046696412782\n",
            "the accurcy is :  0.8528288930832473\n",
            "the accurcy is :  0.8669520017849535\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "array([243.86231644])"
            ]
          },
          "execution_count": 32,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "if not isfile(file_name):\n",
        "  for i in range(50):\n",
        "    X_train ,X_test ,y_train ,y_test =train_test_split(X,y,test_size=0.3)\n",
        "    model = LinearRegression()\n",
        "    model.fit(X_train,y_train)\n",
        "    y_pred =model.predict(X_test)\n",
        "    accurcy = r2_score(y_test,y_pred)\n",
        "    print(\"the accurcy is : \" ,accurcy)\n",
        "    if accurcy>top_acc:\n",
        "      top_acc= accurcy\n",
        "      with open(file_name,'wb') as file:\n",
        "        pickle.dump(model,file)\n",
        "with open(file_name,'rb') as file:\n",
        "  model=pickle.load(file)\n",
        "model.predict([[2.3,5,12]])\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 33,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KW32RZotjw6R",
        "outputId": "09762d14-48f8-4755-d7c1-7cb5314781e7"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "0.8861656208522782"
            ]
          },
          "execution_count": 33,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "top_acc"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sZeGPGXIkb_Y"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "collapsed_sections": [],
      "name": "day6_co2.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.11.5"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
