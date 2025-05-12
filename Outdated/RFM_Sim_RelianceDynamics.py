import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from gspread_dataframe import set_with_dataframe
import random

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = service_account.Credentials.from_service_account_file("XXX.json", scopes=SCOPES) # Input name of credentials
service = build("sheets", "v4", credentials=creds)

SPREADSHEET_ID = "XXX" # Input spreadsheet ID

def plot_ALL_opinions(all_op, num, epsilon, model, network, adj):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_op[:, i])

    plt.xlabel('Iterations')
    plt.ylabel('Opinion')
    plt.title(f'Opinion Evolution in the {model}' , fontweight='bold', pad=36)
    plt.figtext(0.5, 0.9,
                f'Number of individuals: {num}          Epsilon: {epsilon}           Network: {network}           Adjustment Factor: {adj}',
                ha='center', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))
    plt.grid(True)
    plt.show()

def plot_IND_reliance(all_rel, num, epsilon, model, network, adj, target_subject):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_rel[:, i, target_subject])

    plt.xlabel('Iterations')
    plt.ylabel('Reliance')
    plt.title(f'Population\'s Reliance Evolution on Individual {target_subject+1} in the {model}' , fontweight='bold', pad=36)
    plt.figtext(0.5, 0.9,
                f'Number of individuals: {num}          Epsilon: {epsilon}           Network: {network}           Adjustment Factor: {adj}',
                ha='center', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))
    plt.grid(True)
    plt.show()

def plot_ALL_reliance(all_op, num, epsilon, range_inf, model, network, adj):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_op[:, i])

    plt.xlabel('Iterations')
    plt.ylabel('Reliance')
    plt.title(f'Mean of Reliance to {range_inf} Evolution in the {model}' , fontweight='bold', pad=36)
    plt.figtext(0.5, 0.9,
                f'Number of individuals: {num}          Epsilon: {epsilon}           Network: {network}           Adjustment Factor: {adj}',
                ha='center', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))
    plt.grid(True)
    plt.show()

def update_opinions_RFM_OG(A, R, op, num, eps):
    new_opinions = np.copy(op)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]

        relying_individual = [op[j]  * R[i, j] for j in connections if abs(op[i] - op[j]) <= R[i, j] * eps]

        new_opinions[i] = np.mean(relying_individual)

    return new_opinions

def update_opinions_RFM_V2(A, R, op, num, eps, target):
    new_opinions = np.copy(op)
    #influential_reliance_mean = []
    #all_reliance_mean = []

    for i in range(num):
        connections = np.where(A[i] == 1)[0]
        relying_individual = []
        influence = []

        for j in connections:
            if abs(op[i] - op[j]) <= R[i, j] * eps:
                relying_individual += [op[j]  * R[i, j]]
                influence += [R[i, j]]

        new_opinions[i] = np.sum(relying_individual)/np.sum(influence)
        #influential_reliance_mean.append(np.mean(influence))
        #all_reliance_mean.append(np.mean(R[i]))
        if i == target:
            num_influence = str(len(influence))
   
    return new_opinions

def update_opinions_RFM_IOW(A, R, op, num, eps):
    new_opinions = np.copy(op)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]

        relying_individual = [op[j]  * R[i, j] for j in connections if abs(op[i] - op[j]) <= R[i, j] * eps and j != i]
        
        if len(relying_individual) > 0:
            c = len(relying_individual)/len(connections)
            new_opinions[i] = c * np.mean(relying_individual) + (1 - c) * op[i]

    return new_opinions

def update_reliance(A, R, op, num, eps, adj, timestep, target):
    new_R = np.copy(R)
    sudden_increase_individuals = []
    sudden_increase_timesteps = []
    #target_reliance_on_j = ''
    j_reliance_on_target = ''
    j_opinion = ''
    target_affect_j = ''
    j_affect_target = ''
    num_influence = 0

    for i in range(num):
        connections = np.where(A[i] == 1)[0]

        for j in connections:
            if abs(op[i] - op[j]) <= new_R[i, j] * eps:
                if i == target:
                    if timestep > 50 and new_R[i, j] == 0:
                        sudden_increase_individuals.append(j+1)
                        sudden_increase_timesteps.append(timestep)
                        #target_reliance_on_j += f'{R[target, j]} '
                        j_reliance_on_target += f'{R[j, target]} '
                        j_opinion += f'{op[j]} '
                        target_affect_j += f'{A[j, target]} '
                        j_affect_target += f'{A[target, j]} '
                new_R[i, j] += adj
                new_R[i, j] = min(new_R[i, j], 1)
                if i == target:
                    num_influence += 1
            else:
                new_R[i, j] -= adj
                new_R[i, j] = max(new_R[i, j], 0)

    sudden_increase_individuals = ' '.join(map(str, sudden_increase_individuals))
    sudden_increase_timesteps = ' '.join(map(str, sudden_increase_timesteps))

    return new_R, sudden_increase_individuals, sudden_increase_timesteps, j_reliance_on_target, j_opinion, target_affect_j, j_affect_target
#-------------------------------------------------------------------------------

num_individuals = 100
round_op_to_decimal = 3
convergence_boundary = 1*(10**(-round_op_to_decimal))
dbscan = DBSCAN(eps=convergence_boundary, min_samples=2)

A_comp = np.ones((num_individuals, num_individuals))

G_rand = nx.erdos_renyi_graph(num_individuals, 0.5, directed=True)
A_rand = nx.to_numpy_array(G_rand)
np.fill_diagonal(A_rand, 1)

test_cases = [
    #[100, 300, 0.1, 1, 0.1, False],
    #[100, 1000, 0.1, 1, 0.1, False],
    #[100, 300, 0.1, 100, 0.1, False],
    #[100, 300, 0.1, 100, 0.2, False],
    #[100, 300, 0.1, 100, 0.05, False],
    [100, 1000, 0.1, 1, 0.1, False, A_comp],
    #[100, 1000, 0.1, 100, 0.1, False, A_rand],
]

num_rand_structures = 1


#-------------------------------------------------------------------------------

#Construct Complete Network
#A_comp = np.ones((num_individuals, num_individuals))

opinions = np.random.rand(num_individuals)
R = (np.random.rand(num_individuals, num_individuals))

for test in range(len(test_cases)):
    num_iterations_E1 = test_cases[test][0]
    num_iterations_E2 = test_cases[test][1]
    epsilon = test_cases[test][2]
    num_opinions = test_cases[test][3]
    adjustment_factor = test_cases[test][4]

    num_cluster_formations_E2 = []
    stabilize_timestamps_E2 = []
    swapping_err_E2 = []
    runtime_err_E2 = []
    converge_to_E2 = []

    target_trialed = [] # X
    num_affection_E2 = [] # , 2-1
    num_being_affected_E2 = [] # , 2-2
    num_sudden_increases = [] # 1-4, 2-3
    #sudden_increase_prev_num_influence_E2 = [] # 2-3,
    #sudden_increase_num_influence_E2 = [] # 2-4, 
    sudden_increase_individuals_E2 = []
    sudden_increase_timesteps_E2 = [] # 2-1, 3-1
    sudden_increase_target_opinion_E2 = [] # 2-2, 3-2
    #sudden_increase_target_reliance = [] # 2-5,
    sudden_increase_in_opinion_E2 = [] # 2-6, 3-3
    sudden_increase_in_affect_E2 = [] # , 3-4
    sudden_increase_out_affect_E2 = [] # , 3-5
    sudden_increase_reliance_on_target_E2 = [] # 2-7, 3-6

    #opinions = np.random.rand(num_individuals)
    #R = (np.random.rand(num_individuals, num_individuals)) * A_comp
    #if test_cases[test][5]:
     #   np.fill_diagonal(R, 1)

    target_subject1 = 0

    for i in range(num_opinions):
        #initilize common opinion set
        #opinions = np.random.rand(num_individuals)
        #R = (np.random.rand(num_individuals, num_individuals)) * A_comp
        #if test_cases[test][5]:
         #   np.fill_diagonal(R, 1)
        

        #Comment out if necessary
        sum_clusters_RAND = 0
        sum_converge_time_RAND = 0
        sum_swap_err_RAND = 0
        sum_rt_err_RAND = 0
        #Comment out if necessary
        
        #Comment out if necessary
        for k in range(num_rand_structures):
            #G_rand = nx.erdos_renyi_graph(num_individuals, 0.5, directed=True)
            #A_rand = nx.to_numpy_array(G_rand)
            #np.fill_diagonal(A_rand, 1)
        #Comment out if necessary
        #Tab starts here
            converge_time_E2 = 0
            swap_E2 = False

            curr_opinions_E2 = np.copy(opinions)
            all_opinions_E2 = [curr_opinions_E2]
            curr_R_E2 = np.copy(R)
            all_reliance_E2 = [curr_R_E2]
            #all_mean_influential_reliance_E2 = []
            #all_mean_reliance_E2 = []
            first_swap_occurence_E2 = np.array([])
            accumulated_prev_num_influence_E2 = ''
            accumulated_num_influence_E2 = ''
            accumulated_sudden_increase_individuals_E2 = ''
            accumulated_sudden_increase_timesteps_E2 = ''
            accumulated_sudden_increase_target_opinion_E2 = ''
            #accumulated_sudden_increase_target_reliance_E2 = ''
            accumulated_sudden_increase_in_opinion_E2 = ''
            accumulated_sudden_increase_in_affect_E2 = ''
            accumulated_sudden_increase_out_affect_E2 = ''
            accumulated_sudden_increase_reliance_on_target_E2 = ''

            #target_subject1 = random.randint(0, 99)
            #target_subject2 = random.randint(0, 99)
            #while target_subject1 == target_subject2:
            #   target_subject2 = random.randint(0, 99)

            for h in range(1, num_iterations_E2+1):
                new_opinions_E2 = update_opinions_RFM_V2(test_cases[test][6], curr_R_E2, curr_opinions_E2, num_individuals, epsilon, target_subject1)
                
                if not np.array_equal(np.sort(np.round(new_opinions_E2, round_op_to_decimal)), np.sort(np.round(curr_opinions_E2, round_op_to_decimal))):
                    if h > 1:
                        if not np.array_equal(np.sort(np.round(new_opinions_E2, round_op_to_decimal)), np.sort(np.round(prev_opinions_E2, round_op_to_decimal))):
                            converge_time_E2 = h + 1
                        else:
                            first_swap_occurence_E2  = np.copy(new_opinions_E2)
                    else:
                        converge_time_E2 = h + 1
                
                all_opinions_E2.append(new_opinions_E2)
                #all_mean_influential_reliance_E2.append(influential_reliance_mean_E2)
                #all_mean_reliance_E2.append(all_reliance_mean_E2)
                prev_opinions_E2 = curr_opinions_E2
                curr_opinions_E2 = new_opinions_E2

                if isinstance(adjustment_factor, float):
                    new_R_E2, new_trust_E2, new_trust_t_E2, old_in_R_E2, in_opinion_E2, out_affect_E2, in_affect_E2 = update_reliance(test_cases[test][6], curr_R_E2, curr_opinions_E2, num_individuals, epsilon, adjustment_factor, h, target_subject1)
                    all_reliance_E2.append(new_R_E2)
                    curr_R_E2 = new_R_E2
                    if len(new_trust_E2)>0:
                        #accumulated_prev_num_influence_E2 += prev_num_influence + ' '
                        #accumulated_num_influence_E2 += new_num_influence + ' '
                        accumulated_sudden_increase_individuals_E2 += new_trust_E2 + ' '
                        accumulated_sudden_increase_in_affect_E2 += in_affect_E2 + ' '
                        accumulated_sudden_increase_out_affect_E2 += out_affect_E2 + ' '
                        accumulated_sudden_increase_timesteps_E2 += new_trust_t_E2 + ' '
                        accumulated_sudden_increase_target_opinion_E2 += f'{curr_opinions_E2[target_subject1]} '
                        #accumulated_sudden_increase_target_reliance_E2 += old_out_R + ' '
                        accumulated_sudden_increase_in_opinion_E2 += in_opinion_E2 + ' '
                        accumulated_sudden_increase_reliance_on_target_E2 += old_in_R_E2 + ' '
                

            if np.array_equal(first_swap_occurence_E2, curr_opinions_E2) or np.array_equal(first_swap_occurence_E2, prev_opinions_E2):
                swap_E2 = True
            
            sum_clusters_RAND += len(set(dbscan.fit_predict(curr_opinions_E2.reshape(-1, 1))) - {-1})
            sum_converge_time_RAND += converge_time_E2
            if swap_E2:
                sum_swap_err_RAND += 1
            if converge_time_E2 == num_iterations_E2 + 1:
                sum_rt_err_RAND += 1
                
        #Tab ends here

        #Comment out if necessary
        label_list_E2 = dbscan.fit_predict(curr_opinions_E2.reshape(-1, 1))
        length_label_set_E2 = len(set(label_list_E2) - {-1})

        #num_cluster_formations_E2.append(length_label_set_E2)
        converge_to_E2.append(curr_opinions_E2[np.where(label_list_E2 == 0)[0]][0] if length_label_set_E2==1 else -1)

        #stabilize_timestamps_E2.append(converge_time_E2)
        #swapping_err_E2.append(1 if swap_E2 else 0)
        #if converge_time_E2==num_iterations:
        #   print("this")
        #runtime_err_E2.append(1 if converge_time_E2==num_iterations+1 else 0)
        #Comment out if necessary

        #Comment out if necessary
        avg_num_cluster_RAND = sum_clusters_RAND/num_rand_structures
        num_cluster_formations_E2.append(avg_num_cluster_RAND)

        avg_converge_time_RAND = sum_converge_time_RAND/num_rand_structures
        stabilize_timestamps_E2.append(avg_converge_time_RAND)

        swapping_err_E2.append(sum_swap_err_RAND)
        runtime_err_E2.append(sum_rt_err_RAND)

        #sudden_increase_prev_num_influence_E2.append(accumulated_prev_num_influence_E2)
        #sudden_increase_num_influence_E2.append(accumulated_num_influence_E2)
        sudden_increase_individuals_E2.append(accumulated_sudden_increase_individuals_E2)

        num_affection_E2.append(np.sum(test_cases[test][6][target_subject1, :]))
        num_being_affected_E2.append(np.sum(test_cases[test][6][:, target_subject1]))
        sudden_increase_timesteps_E2.append(accumulated_sudden_increase_timesteps_E2)
        sudden_increase_target_opinion_E2.append(accumulated_sudden_increase_target_opinion_E2)
        #sudden_increase_target_reliance.append(accumulated_sudden_increase_target_reliance_E2)
        sudden_increase_in_opinion_E2.append(accumulated_sudden_increase_in_opinion_E2)
        sudden_increase_in_affect_E2.append(accumulated_sudden_increase_in_affect_E2)
        sudden_increase_out_affect_E2.append(accumulated_sudden_increase_out_affect_E2)
        sudden_increase_reliance_on_target_E2.append(accumulated_sudden_increase_reliance_on_target_E2)
        

        target_trialed.append(target_subject1 + 1)
        num_sudden_increases.append(len(accumulated_sudden_increase_individuals_E2.split()))
        #Comment out if necessary

        #all_opinions_E2 = np.array(all_opinions_E2)

        #target_subject1 += 1

        #-------------------------------------------------------------------------------

        print(f"opinion {i}")
        print(avg_num_cluster_RAND)
        print(avg_converge_time_RAND)
        #plot_ALL_opinions(np.copy(all_opinions_E2), num_individuals, epsilon, "RFM", "Complete", adjustment_factor)
        #plot_IND_reliance(np.copy(all_reliance_E2), num_individuals, epsilon, "RFM", "Complete", adjustment_factor, target_subject1)
        #plot_ALL_reliance(np.copy(all_mean_influential_reliance_E2), num_individuals, epsilon, "Influential Connections", "RFM", "Complete", adjustment_factor)
        #plot_ALL_reliance(np.copy(all_mean_reliance_E2), num_individuals, epsilon, "All Connections", "RFM", "Complete", adjustment_factor)
        

        #-------------------------------------------------------------------------------

    data = {
        'Number of Individuals': [num_individuals] * num_opinions,
        'Number of Iterations': [num_iterations_E2] * num_opinions,
        'Confidence Interval': [epsilon] * num_opinions,
        'Convergence': [convergence_boundary] * num_opinions,
        'Opinion Set': [i for i in range(1, num_opinions+1)],
        'Model': ["RFM V2"] * num_opinions,
        'Adjustment Factor': [adjustment_factor] * num_opinions,
        'Network2': ['Complete'] * num_opinions,
        'Number of Arbitrary Network2 Trialled': [num_rand_structures] * num_opinions,
        'Number of Clusters 2': num_cluster_formations_E2,
        'Convergence Time 2': stabilize_timestamps_E2,
        'Opinion Swaps 2': swapping_err_E2,
        'Runtime Errors 2': runtime_err_E2,
        'Converge To 2': converge_to_E2,
        'Target': target_trialed,
        'Sudden Increase individuals': sudden_increase_individuals_E2,
        '2-1 Number of Individual target affect': num_affection_E2,
        '2-2 Number of Individual affect target': num_being_affected_E2,
        '2-3 Number of Sudden Increases': num_sudden_increases,
        '3-1 Sudden Increase timesteps': sudden_increase_timesteps_E2,
        '3-2 Sudden Increase target opinion': sudden_increase_target_opinion_E2,
        #'Sudden Increase Number of Old Influence': sudden_increase_prev_num_influence_E2,
        #'Sudden Increase Number of New Influence': sudden_increase_num_influence_E2,
        #'Target\'s reliance on Sudden Increase individuals': sudden_increase_target_reliance,
        '3-3 Sudden Increase individuals opinion': sudden_increase_in_opinion_E2,
        '3-4 Sudden Increase individuals affect target': sudden_increase_in_affect_E2,
        '3-5 Sudden Increase target affect individuals': sudden_increase_out_affect_E2,
        '3-6 Sudden Increase individuals reliance on target': sudden_increase_reliance_on_target_E2,
    }

    df = pd.DataFrame(data)

    def export_to_sheets(df):
        sheet_service = service.spreadsheets()
        body = {
            'values': [df.columns.tolist()] + df.values.tolist()
        }
        sheet_service.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range= "FINAL FINAL FINAL",  
            valueInputOption="RAW",
            body=body
        ).execute()

    export_to_sheets(df)

    print(sudden_increase_timesteps_E2)

    print(test_cases[test][6])
    if len(sudden_increase_timesteps_E2)>0:
        first = int(sudden_increase_timesteps_E2[0][:sudden_increase_timesteps_E2[0].find(" ")])
        print(first)
        
        np.savetxt("opinions.txt", all_opinions_E2[first-1], delimiter=',', fmt='%.15f')

        np.savetxt("reliances.txt", all_reliance_E2[first-1], delimiter=',', fmt='%.15f')
        
        np.savetxt("connections.txt", test_cases[test][6], delimiter=',', fmt='%.15f')
