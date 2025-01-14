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

def plot_for_reference(all_op, num, epsilon, model, network, swap):
    plt.figure(figsize=(10, 6))
    for i in range(num):
        plt.plot(all_op[:, i])

    plt.xlabel('Iterations')
    plt.ylabel('Opinion')
    plt.title(f'Opinion Evolution in the {model}' , fontweight='bold', pad=36)
    plt.figtext(0.5, 0.9,
                f'Number of individuals: {num}          Epsilon: {epsilon}           Network: {network}           Swap: {swap}',
                ha='center', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='square,pad=0.5'))
    plt.grid(True)
    plt.show()

def count_decimal_places(number):
    str_number = str(number)
    
    if '.' in str_number:
        decimal_part = str_number.split('.')[1]
        #return max(1, len(decimal_part)-1)
        return len(decimal_part)
    else:
        return 0

def add_node_and_edge(G, A):
    G.add_nodes_from(range(num_individuals))
    for i in range(num_individuals):
        for j in range(num_individuals):
            if A[i, j] == 1:
                G.add_edge(i, j)

def update_opinions_BCM_OG(A, op, num, eps):
    new_opinions = np.copy(op)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]
        
        influential_individuals = [op[j] for j in connections if abs(op[i] - op[j]) <= eps]

        new_opinions[i] = np.mean(influential_individuals)

    return new_opinions

def update_opinions_BCM_IOW(A, op, num, eps):
    new_opinions = np.copy(op)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]

        influential_individuals = [op[j] for j in connections if abs(op[i] - op[j]) <= eps and j != i]

        if len(influential_individuals) > 0:
          c = len(influential_individuals)/len(connections)
          new_opinions[i] = c * np.mean(influential_individuals) + (1 - c) * op[i]
      
    return new_opinions

def update_opinions_RFM_OG(A, R, op, num, eps):
    new_opinions = np.copy(op)

    for i in range(num):
        connections = np.where(A[i] == 1)[0]

        relying_individual = [op[j]  * R[i, j] for j in connections if abs(op[i] - op[j]) <= R[i, j] * eps]

        new_opinions[i] = np.mean(relying_individual)

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
                new_R[i, j] += adj * A[i, j]
                new_R[i, j] = min(new_R[i, j], 1)
            else:
                new_R[i, j] -= adj * A[i, j]
                new_R[i, j] = max(new_R[i, j], 0)

    return new_R
#-------------------------------------------------------------------------------

num_individuals = 100
round_op_to_decimal = 3
convergence_boundary = 1*(10**(-round_op_to_decimal))
dbscan = DBSCAN(eps=convergence_boundary, min_samples=2)

num_iterations_E1 = 50
num_iterations_E2 = 300
epsilon = 0.05
num_opinions = 100

num_rand_structures = 100

adjustment_factor = 'N/A'

#-------------------------------------------------------------------------------

num_cluster_formations_E1 = []
stabilize_timestamps_E1 = []
swapping_err_E1 = []
runtime_err_E1 = []
#converge_to_E1 = []

num_cluster_formations_E2 = []
stabilize_timestamps_E2 = []
swapping_err_E2 = []
runtime_err_E2 = []
#converge_to_E2 = []



#Construct Complete Network
A_comp = np.ones((num_individuals, num_individuals))

for i in range(num_opinions):
    #initilize common opinion set
    opinions = np.random.rand(num_individuals)
    

    #BCM - Complete
    converge_time_E1 = 0
    swap_E1 = False

    curr_opinions_E1 = np.copy(opinions)
    #all_opinions_E1 = [curr_opinions_E1]
    first_swap_occurence_E1 = np.array([])
    
    for h in range(1, num_iterations_E1+1):
        new_opinions_E1 = update_opinions_BCM_OG(A_comp, curr_opinions_E1, num_individuals, epsilon)
        
        if not np.array_equal(np.sort(np.round(new_opinions_E1, round_op_to_decimal)), np.sort(np.round(curr_opinions_E1, round_op_to_decimal))):
            if h > 1:
                if not np.array_equal(np.sort(np.round(new_opinions_E1, round_op_to_decimal)), np.sort(np.round(prev_opinions_E1, round_op_to_decimal))):
                    converge_time_E1 = h + 1
                else:
                    first_swap_occurence_E1 = np.copy(new_opinions_E1)
            else:
                converge_time_E1 = h + 1
        
        #all_opinions_E1.append(new_opinions_E1)
        prev_opinions_E1 = curr_opinions_E1
        curr_opinions_E1 = new_opinions_E1
    
    if np.array_equal(first_swap_occurence_E1, curr_opinions_E1) or np.array_equal(first_swap_occurence_E1, prev_opinions_E1):
        swap_E1 = True

    label_list_E1 = dbscan.fit_predict(curr_opinions_E1.reshape(-1, 1))
    length_label_set_E1 = len(set(label_list_E1) - {-1})

    num_cluster_formations_E1.append(length_label_set_E1)
    #converge_to_E1.append(curr_opinions_E1[np.where(label_list_E1 == 0)[0]][0] if length_label_set_E1==1 else -1)

    stabilize_timestamps_E1.append(converge_time_E1)
    swapping_err_E1.append(1 if swap_E1 else 0)
    runtime_err_E1.append(1 if converge_time_E1==num_iterations_E1 else 0)

    #all_opinions_E1 = np.array(all_opinions_E1)


    #BCM - Random Network

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
        first_swap_occurence_E2 = np.array([])
        
        for h in range(1, num_iterations_E2+1):
            new_opinions_E2 = update_opinions_BCM_OG(A_rand, curr_opinions_E2, num_individuals, epsilon)
            
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
        #plot_for_reference(np.copy(all_opinions_E2), num_individuals, epsilon, "RFM", "Complete", swap_E2)
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
    #plot_for_reference(np.copy(all_opinions_E1), num_individuals, epsilon, "BCM", "Complete", swap_E1)
    #lot_for_reference(np.copy(all_opinions_E2), num_individuals, epsilon, "BCM", "Random", swap_E2)

#-------------------------------------------------------------------------------

data = {
    'Number of Individuals': [num_individuals] * num_opinions,
    'Number of Iterations': [num_iterations_E2] * num_opinions,
    'Confidence Interval': [epsilon] * num_opinions,
    'Convergence': [convergence_boundary] * num_opinions,
    'Opinion Set': [i for i in range(1, num_opinions+1)],
    'Model': ["BCM OG"] * num_opinions,
    'Network1': ['Complete'] * num_opinions,
    'Number of Arbitrary Network1 Trialled': [1] * num_opinions,
    'Number of Clusters 1': num_cluster_formations_E1,
    'Convergence Time 1': stabilize_timestamps_E1,
    'Opinion Swaps 1': swapping_err_E1,
    'Runtime Errors 1': runtime_err_E1,
    #'Converge To 1': converge_to_E1,
    'Network2': ["Random"] * num_opinions,
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
        range="BCM",  
        valueInputOption="RAW",
        body=body
    ).execute()

export_to_sheets(df)