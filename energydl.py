import os
from tensorflow.compat.v1.train import summary_iterator
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
from sklearn.linear_model import LinearRegression
from codecarbon import EmissionsTracker
from scipy.optimize import curve_fit
import sys
import select
import time
from termcolor import colored
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _read_events(log_dir):
    """Dumps TensorFlow event files information into a dictionary.
    
    Parameters
    log_dir: (str) The directory of the event files to read

    Returns: (dict) All the information contained on the specified directory
    """
  
    event_dict = {}
    for fit_mode in os.listdir(log_dir):
        if fit_mode not in event_dict:
        event_dict[fit_mode] = {}
        
        for event_file in sorted(os.listdir(os.path.join(log_dir, fit_mode))):
        current_step = max(event_dict[fit_mode].keys()) + 1 if len(event_dict[fit_mode]) != 0 else 0
        event_path = os.path.join(log_dir, fit_mode, event_file)
        pred_path = os.path.join(log_dir, "predictions", fit_mode)

        for event in summary_iterator(event_path):
            for value in event.summary.value:
                if value.tag in ['epoch_loss', 'epoch_accuracy', 'epoch_precision', 'epoch_recall']:
                    if event.step + current_step not in event_dict[fit_mode]:
                        event_dict[fit_mode][event.step + current_step] = {}
                    event_dict[fit_mode][event.step + current_step][value.tag] = tf.compat.v1.make_ndarray(value.tensor)
    return event_dict


def _predict_acc(event_dict, modes, split=["validation"], attr=["accuracy"]):
    """Predicts future correctness values.

    Attributes
    event_dict: (dict) Information read from a TensorFlow event file
    modes: (list) The training modes from all the previous epochs
    split: (list, defaults to [validation]) The training splits whose correctness will be predicted
    attr: (list, defaults to [accuracy]) The attributes that will be predicted

    Returns: (dict) the predictions for every split, attribute and mode specified
    """
    # Hard coded model precomputed apart
    coefs = {
        "validation": 
            {"epoch_accuracy": [0.0505982751, 0.1620413554, 0.2643336892,0.3083330904, 
                                -0.0075728191, -0.0005079859, 0.0083591419, -0.0003340498]},
        "train":
            {"epoch_accuracy": [-0.18286588, 0.01678332, 0.18419629, 0.96881096,
                                -0.02205215, 0.01522404, 0.01805475, -0.01582546]}  # warning 15
    }

    attr = ["epoch_" + a for a in attr]
    predictions = {}
    for s in split:
        predictions[s] = {}
        for a in attr:
            model = coefs[s][a]
            predictions[s][a] = {}
            
            # Estimate the parameter for this dataset
            metrics = [epoch_info[a].item() for epoch_info in event_dict[s].values()]
            dummified_modes = [mode == m for mode in modes for m in ["freeze", "quant"]]
            pred_previous = []
            for i in range(5, len(metrics)):  # warning 5
                input = metrics[i-4:i] + dummified_modes[2*i-2:2*i+2]
                pred_previous.append([sum([inp*c for inp, c in zip(input, model)])])
            #compensation = LinearRegression().fit(pred_previous, metrics[4:])
            compensation = np.mean([gt - p[0] for p, gt in zip(pred_previous, metrics[5:])])  # warning 5
            """ print("metrics", metrics)
            print("pred_previous", pred_previous)
            print("coefs", compensation)#.coef_, compensation.intercept_) """
            #compensation.coef_, compensation.intercept_ = np.array([1.04]), -0.02
            
            # Predict for each mode
            input = metrics[-4:] + dummified_modes[-2:]
            for m in ["base", "freeze", "quant"]:
                predictions[s][a][m] = []
                mode_final = [m == "freeze", m == "quant"]
                input_final = input + mode_final
                # Predict and lag values to make it autoregressive
                pred_raw = []
                for i in range(10):
                    """ if m == "quant":
                      print(input_final) """
                    pred = sum([c*inp for c, inp in zip(model, input_final)])
                    pred_raw.append(pred)
                    #pred = compensation.predict([[pred]])[0]
                    pred = pred + compensation
                    predictions[s][a][m].append(pred)
                    input_final[:4] = [*input_final[1:4], pred]
                    input_final[4:] = [*input_final[6:8], *mode_final]
                """ if m == "quant":
                  print("pred_raw: ", pred_raw)
                  print("pred_post: ", predictions[s][a][m]) """
    return predictions


def _predict_energy(consume, modes):
    """Predicts future energy consumption attributes.
    
    Parameters
    consume: (dict) Information about energy consumption recorded by CodeCarbon
    modes: (list) The training modes from all the previous epochs
    
    Returns: (dict) The predictions for the next epoch for all the attributes and modes
    """

    # Hard coded model precomputed apart
    coefs = {"energy":    {"quant":0.93189, "freeze":0.76765},
             "emissions": {"quant":0.93234, "freeze":0.7681}}
    
    # Get relevant information
    data = {"energy":     {"base": [], "freeze": [], "quant": []},
            "emissions":  {"base": [], "freeze": [], "quant": []}}
    for tag in consume.keys():
        for metric, mode in zip(consume[tag], modes):
            data[tag][mode].append(metric)
    
    # Compute the mean (the best possible prediction)
    data_means = {tag:
                     {mode: np.mean(value) if len(value) != 0 
                      else np.mean(data[tag]["base"])*coefs[tag][mode] for mode, value in data[tag].items() }
                  for tag in data.keys()}
    return data_means    


def _write_events(log_dir, event_dict, acc_predictions, consume, consume_predictions):
    """Writes TensorFlow registered events and predicted ones into an event file.

    Parameters
    log_dir : (str) Directory where the events will be written
    event_dict: (dict) Information read from a TensorFlow event file
    acc_predictions: (dict) Accuracy predictions previously computed for different modes
    consume: (dict) Information about energy consumption recorded by CodeCarbon
    consume_predictions: (dict) Energy predictions previously computed for different modes
    """

    # Remove previously written data
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    # Iterate over the event_dict and write the events to the file
    for fit_mode in event_dict:
        writer = tf.summary.create_file_writer(os.path.join(log_dir, fit_mode))
        with writer.as_default():
        for step in event_dict[fit_mode]:
            for tag in event_dict[fit_mode][step]:
            #value = tf.compat.v1.Summary.Value(tag=tag, tensor=event_dict[fit_mode][step][tag])
            tf.summary.scalar(tag, event_dict[fit_mode][step][tag], step=step)
        writer.close()
    
    # Iterate over the predictions and write the events to the file
    for fit_mode in acc_predictions:
        current_step = max(event_dict[fit_mode]) + 1
        for tag in acc_predictions[fit_mode]:
        for m in acc_predictions[fit_mode][tag]:
            writer = tf.summary.create_file_writer(os.path.join(log_dir, fit_mode + m))
            with writer.as_default():
            for i, v in enumerate(acc_predictions[fit_mode][tag][m]):
                tf.summary.scalar(tag, v, step=current_step + i)
            writer.close()
    
    # Iterate over the consume and write the events to the file
    writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
    with writer.as_default():
        for tag in consume:
        for step in range(len(consume[tag])):
            tf.summary.scalar(tag, consume[tag][step], step=step)
            tf.summary.scalar(tag + "_cumulative", sum(consume[tag][:step+1]), step=step)
    writer.close()
                            
    # Iterate over the consume predictions and write the events to the file
    for tag in consume_predictions:
        current_step = len(consume[tag])
        for m in consume_predictions[tag]:
        writer = tf.summary.create_file_writer(os.path.join(log_dir, "train" + m))
        with writer.as_default():
            for i in range(5):
            tf.summary.scalar(tag, consume_predictions[tag][m], step=current_step+i)
            tf.summary.scalar(tag + '_cumulative', sum(consume[tag]) + (i+1)*consume_predictions[tag][m], step=current_step+i)
        writer.close()


def _advice(event_dict, acc_predictions, consume, consume_predictions, mode, split="validation", energy_importance=0.5, verbose=True):
    """Calculates the best way to continue training and prints it.
    
    Parameters
    event_dict: (dict) Information read from a TensorFlow event file
    acc_predictions: (dict) Accuracy predictions previously computed for different modes
    consume: (dict) Information about energy consumption recorded by CodeCarbon
    consume_predictions: (dict) Energy predictions previously computed for different modes
    mode: (str) The training mode of the last epoch
    split: (str, defaults to validation) The training split whose accuracy will be optimized
    energy_importance: (float, defaults to 0.5) A value between 0 and 1 representing the energy-accuracy tradeoff
    verbose: (bool, defaults to True) Whether to print information about current and predicted attributes, as well as the system's recommendation
    
    Returns: (str)  The mode for next epoch that optimizes the accuracy-energy tradeoff
    """

    # Get relevant information
    accuracy = [epoch["epoch_accuracy"].item() for epoch in event_dict[split].values()]
    score = [acc / en for acc, en in zip(accuracy, np.cumsum(consume["energy"]))]
    total_energy = sum(consume["energy"])
    if verbose:
        acc_col = colored(f"{100*accuracy[-1]:.2f}%", 'red')
        energy_col = colored(f"{sum(consume['energy']):.3g} kW h", 'red')
        mins_col = colored(f"{int(sum(consume['energy'])*600)} minutes")
        print(f"You have an accuracy of {acc_col}, having consumed {energy_col}. That is like letting a light bulb on for {mins_col}")
        print("Continuing for 10 more epochs would give you, for each mode:")
    
    def _score_approx(x, c, b):
        """Function used to fit recorded score and project it to future epochs."""
        return c / (x + b)
    best_advice = [0, None, None]  # score, optimal epoch, mode
    explore_modes = ["base", "freeze", "quant"] if mode == "base" else [mode]

    for m in explore_modes:
        # Get recorded and predicted score and compute optimal epoch
        pred_accuracy = acc_predictions[split]["epoch_accuracy"][m]
        pred_score = score + [acc / (total_energy + i*consume_predictions["energy"][m]) for i, acc in enumerate(pred_accuracy)]
        (c_opt, b_opt), cov = curve_fit(_score_approx, range(len(pred_score)), pred_score, p0=(100000, 1000000))
        epoch_opt = int(np.round(-b_opt + np.sqrt(c_opt / np.tan(energy_importance*np.pi/2))))

        if verbose:
            acc_col = colored(f"{100*pred_accuracy[-1]:.2f}%", 'red')
            energy_col = colored(f"{10*consume_predictions['energy'][m]:.3g} kW h", 'red')
            mins_col = colored(f"{int(10*consume_predictions['energy'][m]*600)} minutes", 'red')
            print(f"- {colored(m, 'blue')}: {acc_col} consuming {energy_col} more (equivalent to {mins_col} of a light bulb),")
            if epoch_opt < len(accuracy):
                print("  but you have already reached your maximum tradeoff.")
            elif len(accuracy) <= epoch_opt < len(pred_score):
                acc_col = colored(f"{100*pred_accuracy[epoch_opt - len(accuracy)]:.1f}%", 'red')
                energy_col = colored(f"{(epoch_opt-len(accuracy)+1) * consume_predictions['energy'][m]:.3g} kW h", 'red')
                mins_col = colored(f"{int((epoch_opt-len(accuracy)+1) * consume_predictions['energy'][m] * 600)} minutes", 'red')
                eps_col = colored(epoch_opt - len(accuracy) + 1, 'red')
                print(f"  but you would only need {eps_col} to reach your maximum tradeoff of {acc_col} accuracy and {energy_col} more (like {mins_col} of light).")
            else:
                print(f"and you wouldn't have reached your maximum tradeoff yet, " +
                      f"estimated at {colored(epoch_opt - len(accuracy) + 1, 'red')} steps more.")

        # Save mode whose optimal score is best
        if pred_score[min(epoch_opt, len(pred_score)-1)] > best_advice[0]:
            best_advice = [pred_score[min(epoch_opt, len(pred_score)-1)], min(epoch_opt, len(pred_score)-1) - len(accuracy) + 1, m]
    
    if verbose:
       print("Our recommendation is to ", end="")
    if best_advice[1] <= len(score):  # Best epoch has already passed
        if verbose:
            print(f"{colored('stop', 'blue')} the training.")
        return "stop"
    else:  # Best epoch is yet to come in some training mode
        if verbose:
            print(f"continue training for {colored(best_advice[1], 'red')} epochs in mode {colored(best_advice[2], 'blue')}.")
        return best_advice[2]


def _keyboard_input(await_time, auto):
    """States possible inputs and waits for them.
    
    Parameters
    await_time: (float) The time this function waits for user input
    auto: (bool) If True, possible actions aren't printed and won't wait for input
    
    Returns: (str) The registered user input, None if not given
    """
    if not auto:
        print(f'If you want to change mode, type "{colored("freeze", "blue")}" or "{colored("quant", "blue")}". Type "{colored("auto", "blue")}" to let the model decide the best mode.')
        print(f'If you want to stop the training, type "{colored("stop", "blue")}"')
        print(f'If you type another thing or do not type anything in {await_time} seconds, the training will continue with the current mode.')
    else:
        await_time = 0.1
    start = time.time()
    while time.time() - start < await_time:
        i,o,e = select.select([sys.stdin],[],[],0.0001)
        for s in i:
            if s == sys.stdin:
                return sys.stdin.readline().strip()
    return None

def energy_aware_train(model, *args, energy_importance=0.5, log_dir="logs", complete_log_dir="logs_complete", checkpoint_path="cp",
                       epochs=50, await_time=10, split=["validation", "train"], start_predicting=6, **kwargs):
    """
    Train a TensorFlow model interactively to allow the user to maximize the accuracy-energy tradeoff.

    Parameters
    model: (tf.keras.Model) Model to be trained
    energy_importance: (float, defaults to 0.5) A value between 0 and 1 representing the energy-accuracy tradeoff
    log_dir: (str, defaults to logs) Directory to save the registered data by TensorFlow
    complete_log_dir: (str, defaults to logs_tb) Directory to save all the training data, including TensorFlow, CodeCarbon and predicted one
    checkpoint_path: (str, defaults to cp) Directory to save the checkpoints of the model during training
    epochs: (int, defaults to 50) Maximum number of epochs to train the model
    await_time: (float, defaults to 10) Time that the sistem will wait for a user input
    split: (list[str], defaults to [validation, train]) Data splits over which predictions are performed, the first one being the one to optimize
    start_predicting: (int, defaults to 6) First epoch in which predictions and advices are done, must be greater than 5
    *args, **kwargs: parameters to pass to model.fit()
    """
    e = 0
    modes = []
    consume = {"energy": [], "emissions": []}
    mode = "base"
    auto = False
    instructions = ""

    while e < epochs:
        #Process input
        new_instructions = _keyboard_input(await_time, auto)
        if new_instructions == "auto":
            auto = True
        else:
          instructions = new_instructions
        if instructions == "freeze":
            for layer in model.layers[:-1]:
              layer.trainable = False
            mode = "freeze"
        elif instructions == "quant":
            weights = model.get_weights()
            keras.mixed_precision.set_global_policy("float16")
            cloned_model = keras.models.clone_model(model)
            cloned_model.set_weights(model.get_weights())
            model = cloned_model
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy', 'Precision', 'Recall'])
            keras.mixed_precision.set_global_policy("float32")
            mode = "quant"
        elif instructions == "stop":
            break
        print(f"You will be training in mode {colored(mode, 'blue')}.")
        modes += [mode]
        
        # Train model
        #tensorboard_process = subprocess.Popen(['tensorboard', '--logdir', complete_log_dir])
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True)
        tracker = EmissionsTracker(log_level="critical")
        tracker.start()
        model.fit(
            *args,
            **kwargs,
            callbacks=[tb_callback, cp_callback]
        )
        tracker.stop()
        consume["energy"].append(tracker._total_energy.kWh)
        consume["emissions"].append(tracker.final_emissions)

        e += 1
        # process collected data
        event_dict = _read_events(log_dir)
        if e >= start_predicting:
            acc_predictions = _predict_acc(event_dict, modes, split)
            consume_predictions = _predict_energy(consume, modes)
            _write_events(complete_log_dir, event_dict, acc_predictions, consume, consume_predictions)
            instructions = _advice(event_dict, acc_predictions, consume, consume_predictions, modes[-1], split[0], energy_importance, verbose=not auto)
        else:
            _write_events(complete_log_dir, event_dict, [], consume, [])  
        #tensorboard_process.terminate()
    return model
