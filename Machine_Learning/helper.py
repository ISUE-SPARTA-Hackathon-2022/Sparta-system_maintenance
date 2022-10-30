import pandas as pd
import numpy as np
import tensorflow as tf
import datetime, zipfile, winsound
import os, codecs, json, random, itertools

from matplotlib import pyplot as plt, image as mpli
from keras.callbacks import History, ModelCheckpoint
from keras.models import Model, load_model
from keras.utils import plot_model as pltmdl
from urllib.request import urlretrieve
from keras.utils import image_dataset_from_directory as IDD
from sklearn.metrics import confusion_matrix


class helper:
  def __init__(self):
    self.evaluation = self.evaluation()
    self.evaluation.comparison_plot = self.evaluation.comparison_plot()
    self.data_preparations = self.data_preparations()
    self.callbacks = self.callbacks()
    self.save_load = self.save_load()
    self.notifications = self.notifications()
    return
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  class evaluation:
    def history_plot(self, history:History or str, save_path:str = None, name:str = "History Plot",
                     specification:str = "all", grid:bool = False, figsize:tuple = (10, 7),
                     plot_style:str = "-", plot_x:str = "epochs"):
      """Create history plot based on Loaded or History object history data. It requires to enter the Model History or the Library Saved File.

      Args:
          history (History or str): History Object or Library Saved File
          save_path (str, optional): Directory Path. Defaults to None.
          name (str, optional): Saved File name. Defaults to "History Plot".
          specification (str, optional): To set data to be plotted. Defaults to "all".
          grid (bool, optional): Enable grid. Defaults to False.
          figsize (tuple, optional): Set Figure Size. Defaults to (10, 7).
          plot_style (str, optional): Set Plot Style. '+-', '+--', 'o-', 'o--', '--'. Defaults to "-".
          plot_x (str, optional): Set Plot X name. Defaults to "epochs".

      Returns:
          Figure: Plot Figure
      """
      _helper = helper()
      try:
        # Check if:
          # "specification" declared are not in history data.
          # "comp_" are not in "specification".
          # "specification" are not all
        # does return report with available specification and comparison
        if specification not in history.history.keys(
        ) and "comp_" not in specification and specification != "all":
          comparison = 'and comp_*(loss, accuracy, etc.)' if "val_loss" in history.history.keys() else ""
          return print(f"The specification is not valid.\nThe available keys are {history.history.keys()} {comparison}")
        epochs =  range(len(history.history['loss']))
        
        # Display All Data in Plot
        if specification == "all":
          pd.DataFrame(history.history).plot(style = plot_style, grid = grid, figsize = figsize, title = name.replace("_", " ").title())
          legend_label = history.history.keys()
          if save_path:
            self.__save(name.replace(" ", "_"), save_path)
            
        # Display the comparisons of stated Key in the plot
        elif "comp_" in specification:
          spe = specification.replace("comp_", "").lower()
          if f"val_{spe}" not in history.history.keys():
            return print(f"Comparison is Unavailable.\nThe available keys are {history.history.keys()}\nNote: to use comparison it requires validation parameter")
          plt.figure(figsize = figsize)
          plt.grid(grid)
          plt.plot(epochs,
                  history.history[spe],
                  plot_style,
                  label = f'Training {spe}')
          plt.plot(epochs,
                  history.history[f'val_{spe}'],
                  plot_style,
                  label = f'Validation {spe}')
          plt.title(f'Comparison of Training and Validation {spe.title()}')
          legend_label = [spe.capitalize(), f"Validation {spe.title()}"]
          if save_path:
            self.__save((name + f"_comp_{spe}").replace(" ", "_"), save_path)
            
        # Display the stated Key in the plot
        elif specification in history.history.keys():
          pd.DataFrame(history.history[specification]).plot(style = plot_style, grid = grid, figsize = figsize, title = name.replace("_", " "))
          if specification == "lr":
            legend_label = ["Learning Rate"]
          elif "val__" in specification:
            legend_label = [specification.replace("val__", "Validation").capitalize()]
          else:
            legend_label = [specification.capitalize()]
          if save_path:
            self.__save((name + "_" + specification).replace(" ", "_"), save_path)
            
        else:
          return print("Invalid Command")
        
        # Plot configuration
        plt.xlabel(plot_x)
        plt.legend(legend_label, loc='center left', bbox_to_anchor = (1, 0))
        
      except Exception as e:
        print(f"Error: {e}")
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def feature_fine_tune_plot(self, model_1_h:History, model_2_h:History,
                              path:str = None,
                              name:str = "Feature Extraction and Fine Tuning Plot",
                              div_name:str = "Start of Fine Tuning",
                              specification:list = ["all"],
                              grid:bool = False,
                              fig_size:tuple = (10, 7), line_style:str = "-",
                              marker_style:str = None):
      """Feature Fine Tune Plot, plots data of two continues model. model a as base and model b as the continuation.

      Args:
          model_1_h (History): History Object
          model_2_h (History): History Object
          path (str, optional): Directory Path. Defaults to None.
          name (str, optional): Save File name. Defaults to "Feature Extraction and Fine Tuning Plot".
          div_name (str, optional): Splitting of two model label. Defaults to "Start of Fine Tuning".
          specification (list, optional): To set data to be plotted. Defaults to ["all"].
          grid (bool, optional): Set Grid. Defaults to False.
          fig_size (tuple, optional): Set Figure Size. Defaults to (10, 7).
          line_style (str, optional): Set Line Style. Defaults to "-".
          marker_style (str, optional): Set Marker Style. Defaults to None.
      """
      try:
        # Set Values Dictionary
        values = {}
        
        if "all" in specification:
          for key in model_1_h.history.keys():
            values[key] = model_1_h.history[key] + model_2_h.history[key]
          
          plt.figure(figsize = fig_size)
          plt.subplot(2,1,1)
          
          for key in model_1_h.history.keys():
            plt.plot(values[key],
                    ls = line_style, marker = marker_style,
                    label = key.replace("_", " ").title())

        else:
          plt.figure(figsize = fig_size)
          plt.subplot(2,1,1)
          for key in specification:
            if not key in model_1_h.history.keys():
              plt.plot([model_1_h.epoch[-1], model_1_h.epoch[-1]],
                  plt.ylim(), lw = 3, ls = "--", label = div_name)
              plt.grid(grid)
              plt.legend(loc='best', bbox_to_anchor = (0, 0))
              return print(f"The Key: '{key}' is from specifications are not included into available keys."
                          + "\nThis are the Available keys: ",
                          list(model_1_h.history.keys()))
            plt.plot(model_1_h.history[key] + model_2_h.history[key],
                    ls = line_style, marker = marker_style,
                    label = key.replace("_", " ").title())
        
        plt.plot([model_1_h.epoch[-1], model_1_h.epoch[-1]],
            plt.ylim(), lw = 3, ls = "--", label = div_name)
        plt.grid(grid)
        plt.legend(loc='best', bbox_to_anchor = (0, 0))
        
        if path:
          self.__save(name.replace(" ", "_") + specification, path)
      except Exception as e:
        print(f"Error: {e}")
        _helper = helper()
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def view_random_image(self, target_dir:str, target_class: str):
      """View random Image From the Directory Displaying also the Class

      Args:
          target_dir (str): Directory to check
          target_class (list): list of classes

      Returns:
          Image: Image and the Class
      """
      _helper = helper()
      
      try:
        if target_dir[-1] != "/":
          target_dir += "/"
        
        target_folder = target_dir + target_class
        
        random_image = np.random.choice(os.listdir(target_folder), 1)
        random_image = random.sample(os.listdir(target_folder), 1)
        
        img = mpli.imread(target_folder + "/" + random_image[0])
        plt.imshow(img)
        plt.title(target_class.replace("_", " ").capitalize())
        plt.axis(False)
        
        print(f"Image Shape: {np.around(img.shape), 2}")
        return
      except Exception as e:
        print(f"Error: {e}")
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def make_confusion_matrix(self, label_true, label_pred, classes:list = None,
                              figsize:tuple = (10, 10), text_size:int = 15, norm:bool = False,
                              path:bool = False, name:str = "Confusion_matrix"):
      # sourcery skip: assign-if-exp, or-if-exp-identity
      """Makes a labelled confusion matrix comparing predictions and ground truth labels.
      If classes is passed, confusion matrix will be labelled, if not, integer class values
      will be used.
      
      Args:
        y_true (Any): Array of truth labels (must be same shape as y_pred).
        y_pred (Any): Array of predicted labels (must be same shape as y_true).
        classes (list, optional): Array of class labels (e.g. string form). If `None`, integer labels are used. Default to None
        figsize (tuple, optional): Size of output figure. Default to (10, 10).
        text_size (int, optional): Size of output figure text. Default to 15.
        norm (bool, optional): normalize values or not. Default to False.
        path (str, optional): directory to save the figure. Default to None.
        name (str, optional): file name. Default to "Confusion_matrix".
      
      Returns:
        A labelled confusion matrix plot comparing y_true and y_pred.
      Example usage:
        make_confusion_matrix(y_true=test_labels, # ground truth test labels
                              y_pred=y_preds, # predicted labels
                              classes=class_names, # array of class label names
                              figsize=(15, 15),
                              text_size=10)
      """
      _helper = helper()
      
      # this Returns Error to the user that Prediction and True Label are not equal
      if len(label_true) != tf.squeeze(label_pred.shape):
        return print(f"Prediction Label and True Label are not in same shape. label_pred = {label_pred.shape}, label_true = {label_true.shape}")
      try:
        # Create the confusion matrix
        cm = confusion_matrix(label_true, label_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
        n_classes = cm.shape[0] # find the number of classes we're dealing with

        # Plot the figure and make it pretty
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
        fig.colorbar(cax)

        # Are there a list of classes?
        if classes:
          labels = classes
        else:
          labels = np.arange(cm.shape[0])
        
        # Label the axes
        ax.set(title = "Confusion Matrix",
              xlabel = "Predicted label",
              ylabel = "True label",
              xticks = np.arange(n_classes), # create enough axis slots for each class
              yticks = np.arange(n_classes), 
              xticklabels = labels, # axes will labeled with class names (if they exist) or ints
              yticklabels = labels)
        
        # Make x-axis labels appear on bottom
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()

        # Rotate xticks for readability & increase font size
        plt.xticks(rotation = 70, fontsize = text_size)
        plt.yticks(fontsize = text_size)
        
        # Set the threshold for different colors
        threshold = (cm.max() + cm.min()) / 2.

        # Plot the text on each cell
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                    horizontalalignment = "center",
                    color = "white" if cm[i, j] > threshold else "black",
                    size = text_size)
          else:
            plt.text(j, i, f"{cm[i, j]}",
                    horizontalalignment = "center",
                    color = "white" if cm[i, j] > threshold else "black",
                    size = text_size)

        # Save the figure to the current working directory
        if path:
          self.__save(name.replace(" ", "_"), path)
        _helper.notifications.complete()
      except Exception as e:
        print(f"Error: {e}")
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def __save(self, name, path):
      # Set Path in Proper way
      if path[-1] != "/":
        path += "/"
      
      os.makedirs(path, exist_ok = True)
      plt.savefig(f"{path}{name}.png")
      return f"{name}.png is saved in {path}"
#------------------------------------------------------------------------------
    class comparison_plot:
      def main(self, history:History or str, specification:str = "all",
               plot_style:str = "-", plt_title:str = "Summary Plot"):
        """Comparison_plot.main used to initiate comparison plot. It requires to enter the Base Model History or the

        Args:
            history (History or str): History Object or Library Saved File
            specification (str, optional): Specifications to display in the plot. Use , as separations for each present key's. Defaults to "all".
            plot_style (str, optional): Plot style.'+-', 'o-', '--', '+--', 'o--'. Defaults to "-".
            plt_title (str, optional): Set Title for the plot. Defaults to "Summary Plot".

        Returns:
            pd.DataFrame: plot in DataFrame format.
        """
        try:
          return pd.DataFrame(history.history if specification == "all"
                              else history.history[specification]).plot(style = plot_style,
                                                                     title = plt_title)
        except Exception as e:
          print(f"Error: {e}")
          _helper = helper()
          _helper.notifications.error()
#------------------------------------------------------------------------------
      def sub(self, history:History or str, plot:pd.DataFrame,
              specification:str = "all", plot_style:str = "-", 
              path:str = None, name:str = "Summary Plot"):
        """Comparison_plot.sub used to add in the comparison plot.

        Args:
            history (History or str): History Object or Library Saved File
            plot (pd.DataFrame): Plot to add in the comparison plot.
            specification (str, optional): Specifications to display in the plot. Use , as separations for each present key's. Defaults to "all".
            plot_style (str, optional): Plot style.'+-', 'o-', '--', '+--', 'o--'. Defaults to "-".
            path (str, optional): Path to save the plot. Defaults to None.
            name (str, optional): Name of the plot. Defaults to "Summary Plot".

        Returns:
            pd.DataFrame: plot in DataFrame format.
        """
        try:
          plot = pd.DataFrame(history.history if specification == "all"
                        else history.history[specification]).plot(ax = plot,
                                                                  style = plot_style)
          self.__save(name, path)
          return plot
        except Exception as e:
          print(f"Error: {e}")
          _helper = helper()
          _helper.notifications.error()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  class data_preparations:
    def data_checking(self, dir:str):
      """Data Checking Checks the Data inside the given Directory.

      Args:
          dir (str): Directory to be evaluated
      """
      try:
        for dirpath, dirnames, filenames in os.walk(dir):
          print(f"There are {len(dirnames)} directories and {len(filenames)} files in {dirpath}")
      except Exception as e:
        _helper = helper()
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def zip_processing(self, url:str, zip_name:str = "file.zip") -> None:
      """Download Zip from the URL and Save.

      Args:
          url (str): Where to Download.
          zip_name (str, optional): File name. Defaults to "file.zip".

      Returns:
          Report: Directory of the Saved Files
      """
      try:
        if ".zip" not in zip_name:
          zip_name += ".zip"
        
        os.makedirs("./resources", exist_ok = True)
        urlretrieve(url, f"./resources/{zip_name}")

        zip_temp = zipfile.ZipFile(f"./resources/{zip_name}")
        zip_temp.extractall("./resources/")
        zip_temp.close()
        return print(f"Downloaded, Extracted and Saved. Directory: ./resources/{zip_name}")
      except Exception as e:
        print(f"Error Occurred: {e}")
        _helper = helper()
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def train_test_dir_setter(self, dir:str,
                              IMG_SIZE:tuple = (224, 224),
                              BATCH: int = 32,
                              CLASS_MODE:str = "binary",
                              SHUFFLE:bool = True):
      """Train and Test Data Setter, From directory of images that are stored in test and train folder it sets back preprocessed images and can return class names.

      Args:
          dir (str): Directory of collective files splitted in train and test folder.
          IMG_SIZE (tuple, optional): Set Image size to be preprocessed. Defaults to (224, 224).
          BATCH (int, optional): Set Batch size of preprocessed images. Defaults to 32.
          CLASS_MODE (str, optional): Set Class to be preprocessed. Binary or Categorical. Defaults to "binary".
          SHUFFLE (bool, optional): Set Shuffle.Default to False.

      Returns:
          Test_IDD_Data, Train_IDD_Data, Evaluation_IDD_Data, CLASS_NAME: Return IDD_Data of train, test, evaluation and/or class names
      """
      _helper = helper()
      try:
        test_data, train_data, evaluation_data = False, False, False
        
        if dir[-1] != "/":
          dir += "/"
        
        for folder in os.scandir(dir):
          if folder.is_dir():
            if folder.name == "test":
              test_data = self.__data_IDD_Setter(dir, "test",
                                                 IMG_SIZE, BATCH, CLASS_MODE, SHUFFLE)
              CLASS_NAME = test_data.class_names
            if folder.name == "train":
              train_data = self.__data_IDD_Setter(dir, "train",
                                                  IMG_SIZE, BATCH, CLASS_MODE, SHUFFLE)
              CLASS_NAME = train_data.class_names
            if folder.name == "evaluation":
              evaluation_data = self.__data_IDD_Setter(dir, "evaluation",
                                                       IMG_SIZE, BATCH, CLASS_MODE, SHUFFLE)
              CLASS_NAME = evaluation_data.class_names
        
        _helper.notifications.preprocessing_data()
        
        print(f"Class Names: {CLASS_NAME}")
        if test_data:
          if train_data:
            if evaluation_data:
              return test_data, train_data, evaluation_data, CLASS_NAME
            return test_data, train_data, CLASS_NAME
          if evaluation_data:
            return test_data, evaluation_data, CLASS_NAME
          return test_data, CLASS_NAME
        if train_data:
          if evaluation_data:
            return train_data, evaluation_data, CLASS_NAME
          return train_data, CLASS_NAME
        if evaluation_data:
          return evaluation_data, CLASS_NAME
        
      except Exception as e:
        return print(f"Error Occurred: {e}")
      _helper.notifications.error()
#------------------------------------------------------------------------------
    def __data_IDD_Setter(self, dir, type, IMG_SIZE, BATCH, CLASS_MODE, SHUFFLE):
      try:
        print(f"{type.capitalize()} Data: ")
        return IDD(directory = dir + type + "/",
                   image_size = IMG_SIZE,
                   batch_size = BATCH,
                   label_mode = CLASS_MODE,
                   shuffle = SHUFFLE)
      except Exception as e:
        print(f"Error Occurred: {e}")
        _helper = helper()
        _helper.notifications.error()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  class callbacks:
    def tensorboard_callback(self, dir_name:str, model:str):
      """Set Tensor Board Log to be used in fit function model

      Args:
          dir_name (str): Where to save.
          model (str): Model that being log.

      Returns:
          tensorboard_callback: Returns Tensorboard to callback inside the fit function of model.
      """
      try:
        log_dir = dir_name + "/" + model + "/" + datetime.datetime.now().strftime("%Y_%m_%d__%H_%M") +"_log"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir)
        print(f"Saving TensorBoard Log Files to: {log_dir}")
        return tensorboard_callback
      except Exception as e:
        print(f"Error {e}")
        _helper = helper()
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def checkpoint_callback(self, dir:str, name:str, monitor: str = "loss",
                            sbo: bool = False, swo: bool = False, sf: str = "epoch"):
      """Set checkpoint callback to be used in fit function of model

      Args:
          dir (str): Where to save.
          name (str): File name.
          monitor (str, optional): Key to monitor which to saved as checkpoint. Defaults to "loss".
          sbo (bool, optional): Save Best Only. Defaults to False.
          swo (bool, optional): Save Weight Only. Defaults to False.
          sf (str, optional): Frequency of saving. Defaults to "epoch".

      Returns:
          checkpoint_callback: Returns Checkpoint to callback inside the fit function of model.
      """
      try:
        if dir[-1] != '/':
          dir += '/'
          
        if name[-5:] != '.ckpt':
          name += '.ckpt'
        
        cp_dir = dir + "checkpoint/" + name
        
        return ModelCheckpoint(filepath = cp_dir,
                                monitor = monitor,
                                save_best_only = sbo,
                                save_weights_only = swo,
                                save_freq = sf), cp_dir
      except Exception as e:
        print(f"Error: {e}")
        _helper = helper()
        _helper.notifications.error()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  class save_load:
    def save(self, model:Model = None, history:History = None,
             path:str = None, name:str = "Model"):
      """Save Model and/or History.

      Args:
          model (Model, optional): model data. Defaults to None.
          history (History, optional): history data. Defaults to None.
          path (str, optional): Directory to save from. Defaults to None.
          name (str, optional): File name. Defaults to "Model".
      """
      _helper = helper()
      
      if path[-1] != "/":
        path += "/"
      
      dirname_model = f"{path + name}_{datetime.datetime.now().strftime('%Y_%m_%d - %H')}.h5"
      dirname_history = f"{path + name}_{datetime.datetime.now().strftime('%Y_%m_%d - %H')}_history.json"
      dirname_diagram = f"{path + name}_diagram.png"
      new_hist = {}
      
      try:
        os.makedirs(path, exist_ok = True)
        
        model.save(dirname_model)
        
        if history:
          for key in list(history.history.keys()):
            new_hist[key] = history.history[key]
            if type(history.history[key]) == np.ndarray:
              new_hist[key] = history.history[key].tolist()
            elif type(history.history[key]) == list:
              if type(history.history[key][0]) == np.float64 or type(history.history[key][0]) == np.float32:
                new_hist[key] = list(map(float, history.history[key]))
        
        with codecs.open(dirname_history, "w", encoding = "utf-8") as file:
          json.dump(new_hist, file, separators = (",", ":"), sort_keys = True, indent = 4)
        
        if model:
          pltmdl(model = model,
                show_shapes =True,
                show_layer_activations = True,
                to_file = dirname_diagram)
        if model:
          print(f"Model is saved to {dirname_model} and {dirname_diagram}")
        if history:
          print(f"History is saved to {dirname_history}")
        
        _helper.notifications.file_saving()
      except Exception as e:
        print(f"Error Occurred {e}")
        _helper.notifications.error()
#------------------------------------------------------------------------------
    def load(self, history_path:str = None, model_path:str = None):
      """Load Model Data or History Data

      Args:
          history_path (str, optional): Directory path of the Model. Defaults to None.
          model_path (str, optional): Directory path of the History. Defaults to None.

      Returns:
          _type_: _description_
      """
      try:
        if not model_path and not history_path:
          return print("There is no History and/or Model to be saved")
        
        if model_path:
          m = load_model(model_path)
        
        if history_path:
          with codecs.open(history_path, 'r', encode = 'utf-8') as file:
            h = json.loads(file.read())
        
        if model_path and history_path:
          print("Model and History has been loaded")
          m.summary()
          return h, m
        elif model_path:
          print("Model has been loaded")
          m.summary()
          return m
        elif history_path:
          print("History has been loaded")
          return h
      except Exception as e:
        print(f"Error: {e}")
        a = helper()
        a.notifications.error()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
  class notifications:
    def notification(self, duration:int = 1000, # milliseconds 
                    freq:int = 440  # Hz
                    ):
      winsound.Beep(duration, freq)
#------------------------------------------------------------------------------
    def model_callback_notification(self):
      self.__notification_set("model_callback")
#------------------------------------------------------------------------------
    def file_saving(self):
      self.__notification_set("file_save")
#------------------------------------------------------------------------------
    def file_loading(self):
      self.__notification_set("file_load")
#------------------------------------------------------------------------------
    def preprocessing_data(self):
      self.__notification_set("preprocessing_data")
#------------------------------------------------------------------------------
    def complete(self):
      self.__notification_set("complete")
#------------------------------------------------------------------------------
    def error(self):
      self.__notification_set("error")
#------------------------------------------------------------------------------
    def __notification_set(self, type: str, ):
      dir = f"./notification/{type}/"
      sound = np.random.choice(os.listdir(dir), 1)
      sound = random.sample(os.listdir(dir), 1)

      winsound.PlaySound(sound = dir + sound[0],
                          flags = winsound.SND_NOSTOP)
