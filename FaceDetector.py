{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4bb4c0dc-958c-4f4c-9969-0a94df179815",
   "metadata": {},
   "source": [
    "### Image Classification using SVM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7b04d5ad-e32d-4485-b500-4eb4a652440e",
   "metadata": {},
   "source": [
    "#### Import necessary libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e8d692b8-01a2-4d4c-bed5-0fa511a67e42",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\anaconda1\\Lib\\site-packages\\paramiko\\transport.py:219: CryptographyDeprecationWarning: Blowfish has been deprecated and will be removed in a future release\n",
      "  \"class\": algorithms.Blowfish,\n"
     ]
    }
   ],
   "source": [
    "from PIL import Image\n",
    "import os\n",
    "import re\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "\n",
    "import pylab as pl\n",
    "from matplotlib import pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import GridSearchCV\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.svm import SVC\n",
    "\n",
    "from skimage.feature import hog"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64b4a4f7-2b7d-429b-8dae-9b5a92c8eb51",
   "metadata": {},
   "source": [
    "#### Construct datatset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "06912d19-09e1-456c-82f3-e9908d639aa4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import zipfile"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "99122c70-2ff4-453c-b370-ee7d854ad502",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_zip_path = 'testing_photos.zip'\n",
    "train_zip_path = 'training_photos.zip'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "7bf4227c-5e08-4352-8d54-ada6db1f33c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "with zipfile.ZipFile(train_zip_path, 'r') as train_zip:\n",
    "    train_zip.extractall()\n",
    "\n",
    "with zipfile.ZipFile(test_zip_path, 'r') as test_zip:\n",
    "    test_zip.extractall()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f5351adb-f2ba-478e-bc8d-3209509702a2",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_folder = 'a1_TrainingPhotos/'\n",
    "test_folder = 'a2_NewPhotos/'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "91351949-84d2-4ca5-aab9-92e8119e2300",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = []\n",
    "Y_train = []\n",
    "X_test = []\n",
    "Y_test = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e1164c0e-4479-490c-8b8d-09d078cdd444",
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_images(folder):\n",
    "    X = []\n",
    "    Y = []\n",
    "    for filename in os.listdir(folder):\n",
    "        img = Image.open(os.path.join(folder, filename))\n",
    "        img = img.resize([32, 64])  # Resize image to a fixed size\n",
    "        img = img.convert('L')  # Convert image to grayscale\n",
    "        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2), visualize=True)  # Obtain HOG features\n",
    "        X.append(fd)\n",
    "\n",
    "        # Obtain labels from file name\n",
    "        if re.match(\"Amir*\", filename):\n",
    "            Yt = 0\n",
    "        elif re.match(\"Jaya*\", filename):\n",
    "            Yt = 1\n",
    "        elif re.match(\"Hir*\", filename):\n",
    "            Yt = 2\n",
    "        elif re.match(\"Anupam*\", filename):\n",
    "            Yt = 3\n",
    "        else:\n",
    "            Yt = 4\n",
    "\n",
    "        Y.append(Yt)\n",
    "    \n",
    "    return X, Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "8a013b18-d94c-4981-9ad1-27f93c8652bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, Y_train = process_images(train_folder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "08b48a13-aec3-414d-a76c-d1f84ea61a9d",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test, Y_test = process_images(test_folder)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "3ae9450d-bf83-498d-86be-6547b67d92ba",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in X_test:\n",
    "    X_train.append(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "b3fc13fd-9c3b-4b2f-b541-2eaa264b9742",
   "metadata": {},
   "outputs": [],
   "source": [
    "for i in Y_test:\n",
    "    Y_train.append(i)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc704e9f-97f5-4c72-8616-c07dd45f7a34",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "59db1852-a38e-408e-9c2a-f1a58e56b166",
   "metadata": {},
   "outputs": [],
   "source": [
    "target_names=['Aamir', 'Jaya', 'Hritik', 'Anupam']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "46db53de-190d-44ad-9fae-aa61b2e622f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "5fa977b5-ae5d-4d77-a3d3-8629e593902e",
   "metadata": {},
   "outputs": [],
   "source": [
    "x_train = np.array(x_train)\n",
    "x_test = np.array(x_test)\n",
    "y_train = np.array(y_train)\n",
    "y_test = np.array(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "7f0aa05c-6156-4447-af31-c832d3a2b1fd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(429, 3780)"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "76a1caf2-523b-4849-8316-0fb80dc045b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 2, 0, 2, 0, 1, 2, 2, 3, 3, 1, 0, 1, 2, 0, 0, 0, 3, 1, 0, 1,\n",
       "       1, 1, 0, 2, 2, 1, 3, 3, 1, 0, 1, 0, 1, 1, 2, 1, 2, 0, 2, 0, 2, 3,\n",
       "       0, 2, 0, 2, 2, 0, 0, 3, 1, 0, 0, 3, 0, 0, 1, 0, 2, 2, 3, 1, 0, 1,\n",
       "       1, 0, 1, 2, 3, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 2, 0, 1, 0, 2, 2, 0,\n",
       "       3, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, 2, 0, 0, 1, 1, 1,\n",
       "       1, 1, 1, 0, 1, 0, 3, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 2, 1, 1,\n",
       "       1, 0, 0, 0, 0, 0, 2, 0, 1, 1, 1, 0, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2,\n",
       "       0, 0, 2, 1, 0, 0, 2, 2, 2, 1, 0, 2, 1, 0, 1, 0, 2, 0, 1, 0, 2, 2,\n",
       "       0, 1, 0, 0, 3, 2, 0, 1, 1, 0, 0, 0, 3, 2, 2, 1, 1, 1, 1, 0, 1, 3,\n",
       "       1, 3, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 2, 0, 0, 2, 2, 2, 2, 0, 1,\n",
       "       3, 1, 2, 0, 2, 0, 0, 2, 3, 0, 0, 1, 3, 1, 2, 0, 1, 2, 1, 1, 0, 1,\n",
       "       2, 0, 1, 0, 1, 0, 2, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 3, 0, 0,\n",
       "       3, 1, 2, 1, 0, 0, 1, 1, 0, 2, 2, 2, 1, 0, 2, 1, 0, 3, 1, 3, 1, 0,\n",
       "       0, 2, 3, 0, 0, 0, 2, 2, 0, 1, 0, 1, 3, 2, 1, 2, 0, 1, 1, 0, 0, 0,\n",
       "       0, 1, 0, 2, 1, 0, 2, 0, 0, 0, 1, 1, 1, 3, 1, 2, 1, 1, 2, 3, 3, 3,\n",
       "       2, 1, 2, 1, 1, 2, 2, 2, 1, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 1, 0, 1,\n",
       "       2, 0, 1, 0, 0, 0, 1, 2, 1, 1, 1, 2, 0, 2, 1, 0, 2, 2, 3, 0, 2, 0,\n",
       "       1, 2, 0, 2, 0, 0, 2, 1, 1, 2, 3, 0, 2, 0, 1, 0, 1, 0, 3, 1, 0, 1,\n",
       "       0, 0, 1, 0, 2, 3, 2, 0, 2, 1, 0, 2, 0, 1, 1, 1, 2, 2, 0, 0, 2, 0,\n",
       "       1, 2, 3, 1, 0, 0, 0, 0, 2, 1, 0])"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e2c767f4-38ad-4609-9ff4-9f09969a665d",
   "metadata": {},
   "source": [
    "#### Set parameters and define model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "f01bc375-c319-4a88-9fec-1cf61fb67115",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVC(C=1000.0, class_weight='balanced', gamma=0.005, probability=True)\n"
     ]
    }
   ],
   "source": [
    "param_grid = {\n",
    "         'C': [1e3, 5e3, 1e4, 5e4, 1e5],\n",
    "          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],\n",
    "          }\n",
    "clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced',probability=True), param_grid)\n",
    "clf = clf.fit(x_train, y_train)\n",
    "\n",
    "print(clf.best_estimator_)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "e21f7a22-80cd-4a8e-afda-1ba3992eb2be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['c2_FR_svm_classifier.pkl']"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import joblib\n",
    "joblib.dump(clf.best_estimator_, 'c2_FR_svm_classifier.pkl', compress = 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d3956409-f860-496a-a6e0-88da5501f86b",
   "metadata": {},
   "source": [
    "#### Test model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "892339df-ac47-4f47-a77e-9a955cf13178",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 2, 0, 1, 0, 1, 1, 0, 3, 2, 1, 1, 0, 3, 1, 2, 0, 0, 3, 0, 0,\n",
       "       0, 2, 2, 0, 2, 0, 0, 1, 2, 0, 3, 2, 1, 1, 0, 0, 2, 0, 0, 2, 1, 1,\n",
       "       2, 0, 0, 3, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 2, 0, 0, 2, 0, 1, 2, 1,\n",
       "       0, 2, 2, 2, 0, 3, 1, 1, 0, 1, 1, 1, 1, 2, 2, 0, 1, 0, 0, 0, 0, 0,\n",
       "       2, 0, 1, 0, 1, 2, 0, 0, 1, 1, 2, 0, 0, 1, 0, 0, 2, 2, 0, 1])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = clf.predict(x_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "ef3b3748-a37a-44fc-99f8-8caf28eebce8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.33      0.50      0.40         2\n",
      "           1       1.00      1.00      1.00         1\n",
      "           2       0.50      1.00      0.67         1\n",
      "           3       0.00      0.00      0.00         2\n",
      "\n",
      "    accuracy                           0.50         6\n",
      "   macro avg       0.46      0.62      0.52         6\n",
      "weighted avg       0.36      0.50      0.41         6\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\anaconda1\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "C:\\anaconda1\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n",
      "C:\\anaconda1\\Lib\\site-packages\\sklearn\\metrics\\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.\n",
      "  _warn_prf(average, modifier, msg_start, len(result))\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(Y_test, Y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a771effd-d22b-41cc-8ec5-131e8be433e5",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
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
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
