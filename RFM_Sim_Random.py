import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from gspread_dataframe import set_with_dataframe

SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
creds = service_account.Credentials.from_service_account_file("XXX.json", scopes=SCOPES) # Input name of credentials
service = build("sheets", "v4", credentials=creds)

SPREADSHEET_ID = "XXX" # Input spreadsheet ID

def plot_for_reference(all_op, num, epsilon, model, network, adj):
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

def plot_reliance_for_reference(all_rel, num, epsilon, model, network, adj):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_rel[:, i, target_subject-1])

    plt.xlabel('Iterations')
    plt.ylabel('Reliance')
    plt.title(f'Population\'s Reliance Evolution on Individual {target_subject} in the {model}' , fontweight='bold', pad=36)
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

def update_opinions_RFM_V2(A, R, op, num, eps):
    new_opinions = np.copy(op)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]
        relying_individual = []
        influence = []

        for j in connections:
            if abs(op[i] - op[j]) <= R[i, j] * eps:
                relying_individual += [op[j]  * R[i, j]]
                influence += [R[i, j]]

        new_opinions[i] = np.sum(relying_individual)/np.sum(influence)

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

def update_reliance(A, R, op, num, eps, adj):
    new_R = np.copy(R)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]

        for j in connections:
            if abs(op[i] - op[j]) <= new_R[i, j] * eps:
                new_R[i, j] += adj
                new_R[i, j] = min(new_R[i, j], 1)
            else:
                new_R[i, j] -= adj
                new_R[i, j] = max(new_R[i, j], 0)

    return new_R
#-------------------------------------------------------------------------------

num_individuals = 100
round_op_to_decimal = 3
convergence_boundary = 1*(10**(-round_op_to_decimal))
dbscan = DBSCAN(eps=convergence_boundary, min_samples=2)
target_subject = 1

test_cases = [
    #[100, 100, 0.05, 30, 'N/A', False],
    #[100, 1000, 0.05, 30, 'N/A', False],
    #[100, 1000, 0.1, 30, 'N/A', False],
    #[100, 100, 0.05, 100, 'N/A', True],
    #[100, 100, 0.1, 100, 'N/A', True],
    [100, 1000, 0.1, 30, 0.05, False],
    #[100, 1000, 0.1, 30, 0.1, False],
    #[100, 300, 0.1, 100, 0.2, False]
]

num_rand_structures = 30


#-------------------------------------------------------------------------------

#Construct Complete Network
A_comp = np.ones((num_individuals, num_individuals))

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
    #converge_to_E2 = []

    for i in range(num_opinions):
        #initilize common opinion set
        opinions = np.random.rand(num_individuals)
        R = (np.random.rand(num_individuals, num_individuals) * 0.5) * A_comp
        if test_cases[test][5]:
            np.fill_diagonal(R, 1)
        

        #Comment out if necessary
        sum_clusters_RAND = 0
        sum_converge_time_RAND = 0
        sum_swap_err_RAND = 0
        sum_rt_err_RAND = 0
        #Comment out if necessary
        
        #Comment out if necessary
        for k in range(num_rand_structures):
            G_rand = nx.erdos_renyi_graph(num_individuals, 0.5, directed=True)
            A_rand = nx.to_numpy_array(G_rand)
            np.fill_diagonal(A_rand, 1)
        #Comment out if necessary
        #Tab starts here
            converge_time_E2 = 0
            swap_E2 = False

            curr_opinions_E2 = np.copy(opinions)
            #all_opinions_E2 = [curr_opinions_E2]
            curr_R_E2 = np.copy(R)
            #all_reliance_E2 = [curr_R_E2]
            first_swap_occurence_E2 = np.array([])
            
            for h in range(1, num_iterations_E2+1):
                new_opinions_E2 = update_opinions_RFM_V2(A_rand, curr_R_E2, curr_opinions_E2, num_individuals, epsilon)
                
                if not np.array_equal(np.sort(np.round(new_opinions_E2, round_op_to_decimal)), np.sort(np.round(curr_opinions_E2, round_op_to_decimal))):
                    if h > 1:
                        if not np.array_equal(np.sort(np.round(new_opinions_E2, round_op_to_decimal)), np.sort(np.round(prev_opinions_E2, round_op_to_decimal))):
                            converge_time_E2 = h + 1
                        else:
                            first_swap_occurence_E2  = np.copy(new_opinions_E2)
                    else:
                        converge_time_E2 = h + 1
                
                #all_opinions_E2.append(new_opinions_E2)
                prev_opinions_E2 = curr_opinions_E2
                curr_opinions_E2 = new_opinions_E2

                if isinstance(adjustment_factor, float):
                    new_R_E2 = update_reliance(A_rand, curr_R_E2, curr_opinions_E2, num_individuals, epsilon, adjustment_factor)
                    #all_reliance_E2.append(new_R_E2)
                    curr_R_E2 = new_R_E2

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
        #label_list_E2 = dbscan.fit_predict(curr_opinions_E2.reshape(-1, 1))
        #length_label_set_E2 = len(set(label_list_E2) - {-1})

        #num_cluster_formations_E2.append(length_label_set_E2)
        #converge_to_E2.append(curr_opinions_E2[np.where(label_list_E2 == 0)[0]][0] if length_label_set_E2==1 else -1)

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
        #Comment out if necessary

        #all_opinions_E2 = np.array(all_opinions_E2)

        #-------------------------------------------------------------------------------

        print(f"opinion {i}")
        print(avg_num_cluster_RAND)
        print(avg_converge_time_RAND)
        #plot_for_reference(np.copy(all_opinions_E2), num_individuals, epsilon, "RFM", "Complete", adjustment_factor)
        #plot_reliance_for_reference(np.copy(all_reliance_E2), num_individuals, epsilon, "RFM", "Complete", adjustment_factor)
        

        #-------------------------------------------------------------------------------

    data = {
        'Number of Individuals': [num_individuals] * num_opinions,
        'Number of Iterations': [num_iterations_E2] * num_opinions,
        'Confidence Interval': [epsilon] * num_opinions,
        'Convergence': [convergence_boundary] * num_opinions,
        'Opinion Set': [i for i in range(1, num_opinions+1)],
        'Model': ["RFM V2"] * num_opinions,
        'Adjustment Factor': [adjustment_factor] * num_opinions,
        'Network2': ['Random'] * num_opinions,
        'Number of Arbitrary Network2 Trialled': [num_rand_structures] * num_opinions,
        'Number of Clusters 2': num_cluster_formations_E2,
        'Convergence Time 2': stabilize_timestamps_E2,
        'Opinion Swaps 2': swapping_err_E2,
        'Runtime Errors 2': runtime_err_E2,
        #'Converge To 2': converge_to_E2
    }

    df = pd.DataFrame(data)

    def export_to_sheets(df):
        sheet_service = service.spreadsheets()
        body = {
            'values': [df.columns.tolist()] + df.values.tolist()
        }
        sheet_service.values().append(
            spreadsheetId=SPREADSHEET_ID,
            range="RFMV2 Random",  
            valueInputOption="RAW",
            body=body
        ).execute()

    export_to_sheets(df)