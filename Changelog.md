Used PyLint to make code more in line with PEP8 python code style, e.g:
    - line lengths
    - spacing neatness
    - ordering imports
    - docstrings
    - module strings
    - string formatting
    - raise more specific exceptions
    - consistent x_y instead of xY case everywhere
    - 
-> not really any functional changes from this, just code appearance.

other general changes:
- Made the arg parser not a function - seemed unnecessary extra lines.
- added batch size as argument (has quite a bit effect on time/size so good to have flexibility for gorfrog/tacocat/different signals.. to do different things)
- emd_calc.py deleted, functions instead moved elsewhere (e.g. utils/torch_distances) and made more consistent with the rest of the code using torch.tensors not numpy.
- added a plot_kinematics script, which separates that part out from write_files. Idea is that for paper we might want to rerun the plotting a lot to tweak aesthetics, and shouldn't need to tie that functionality in with the slower ntuple processing, better 'style' to separate modules for distinct tasks.
- generally reordered some pieces (e.g. related to loading in large objects) to minimise the number of big objects open in memory at once. 
- tried to add a lot of delete statements as soon as large objects are not needed anymore. 
- include seb's changes to add EMD and the k-fold setting stuff.
- includes the cut description in names of things for bookkeeping (probably not helpful for LQs but could be for other studies)

big changes to specific files:

write_files
    - made plot of event weights controlled with an argument.
    - remove kinematics plotting stuff -> now in plot_kinematics script that can be run after if needed reads in the h5s.

calc_distances
    - moved distance metric function to utils alongside other related functions.
    - moved the batch distance calc into a function that we call for sigsig/sigbkg/bkgkbg to reduce repitition.
    - encorporate EMD option.
    - when we make the distance metric plot, this was originally done over the first few events in the ntuples, but to better ensure we are sampling the whole MET slices range I changed it to plot a specified fraction of events so e.g. if you said 0.1 it would use every 10th event. I checked this gives much better agreement with the distance metric plots you would get with the entire set of events. 

linking_length
    - decided to re-add the sigsig_eff functionality option, alongisde the edge_frac option. the choice is controlled in the config files. 
    - I tried to make it less specific that you'd want to be using the sig-sig eff specifically, so it is called '*target_eff' instead.
    - flag of whether we want to look at same_class - just true if doing embeddingnet for now.
    - processed the distance values into histogram objects early on, for plot making. This meant that we could delete big objects like the weights sooner.
    - eliminated the Flip option - which currently filled 2 roles of representing which species was the friendliest and whether we wanted to make a friend/enemy graph. I split this into 2 seperate ideas. I figured that flip would always be set manuall based on whether your sig-sig were closer than bkg-bkg or not, so we could just check which case we have from the distribution means automatically and set a is_signal_closest flag to fill this role in hopefully an easier to understand way. for the edge_frac option we don't care about this anyway.
    - added a 'friend_graph' option to the config which would be true for friend/ false for enemy. This gets used in your find_threshold_edge_frac function.
    - made it so if you use your edge_frac option we plot the ROC curve for sigsig vs sigbkg and bkgbkg vs sigbkg as you already had it. Then the efficiency option: withe same-class embedding net we do that too, otherwise it plots the efficiency of your friend class vs the other two (so e.g. for the LQ it should do ss v bb and ss v sb like we had originally; but for HHH it would do bb vs ss and bb vs sb). I was undecided about the best approach for this tbh, I coded it this way and then thought actually it might be better to always do ss vs sb and bb vs sb since we will always want to minimise the sb connections but I haven't changed it...

utils/adj_mat
- rm the plotting stuff from data_loader now I moved it to the plot_kinematics script. Data loader also reads in the info to set the fold, supports HHH/LQ/stau.
- seb's fold setting stuff.

utils/gcn_layer
- fix bug in batch norm use.

utils/graph_definition
- added a version of find_threshold_edge_frac which makes very finely binned histograms and scans through them instead of sorting/scanning the tensor itself. This is substantially less memory intensive, and with the large number of bins I have hardcoded the output values are the same to a few decimal places. your function is still there with _continuous in the name, in case.

utils/misc
- added function to make a filename-friendly string from your cut dictionary.
- get_kinematics no longer has the string laTeX labels in too (all the variable bookkeeping will be together in utils/variables.py, avoids the labels being defined multiple times etc).
- get_batched_distances supports the sampling fraction thing I mentioned earlier to get the nice evenly distributed subset for plotting the distances.
- Seb's deterministic fold function. 

utils/performance
- be more generic by renaming 'sig' 'bkg' to 'target' thing to connect and 'reject' thing to avoid connecting, replace the 'flip' with 'is_target_closest'.

utils/plotting
- move a few things into functions to avoid repetition e.g. saving figures in a list of formats, getting text labels/legends for plots based on the signal/variables, drawing a list of histograms, configuring axes and drawing the text/legend onto the figures.
- imported the mplhep style. does a couple of nice things like making text bigger, tbc if we want to keep this on though.
- add a function that allows us to save all the data/metadata for a figure so that it could be replotted with different aesthetics without rerunning big pieces of code, or incase we lose the inputs etc. 
- a couple of variant functions for e.g. linking length plots either with the tensors or pre-defined histograms as inputs. 

utils/variables
- stores all the metadata for kinematic variables (could also add distances etc here too) for plotting config. at the moment this is just the axis labels, but the goal would be to define optimal binning, ranges, could add tags for which variables belong to which sets/signals etc.. 

ml_<>.yaml
- added flags for friend_graph, targettarget_eff (if you want to use that not edge_frac). Moved n_folds to the user config, becuase we need that in earlier scripts now where we aren't using the ml config. 

user_<>.yaml
- added the n_folds flag.
- added Seb's 'run_with_cuda' flag.



potential todos
- check printing out full paths not just config ones
- if no signal mass will anything crash?
- ml config class
- asserts into config classes
- refactor utils into more helpfully named files