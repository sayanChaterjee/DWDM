import time
from itertools import combinations
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class ProbabilisticApriori:
    def __init__(self, min_support: float, min_confidence: float):
        """
        Initializes the algorithm with probabilistic thresholds.
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {} 
        
    def _generate_candidates(self, prev_frequent_itemsets: dict, k: int) -> set:
        """
        Generates candidate k-itemsets by joining frequent (k-1)-itemsets.
        Enforces the downward closure property via subset pruning.
        """
        candidates = set()
        prev_items = list(prev_frequent_itemsets.keys())
        length = len(prev_items)
        
        for i in range(length):
            for j in range(i + 1, length):
                itemset1 = sorted(list(prev_items[i]))
                itemset2 = sorted(list(prev_items[j]))
                
                if itemset1[:k-2] == itemset2[:k-2]:
                    candidate = frozenset(prev_items[i] | prev_items[j])
                    
                    subsets_are_frequent = True
                    for subset in combinations(candidate, k-1):
                        if frozenset(subset) not in prev_frequent_itemsets:
                            subsets_are_frequent = False
                            break
                    
                    if subsets_are_frequent:
                        candidates.add(candidate)
        return candidates

    def _calculate_expected_support(self, candidates: set, transactions: list) -> dict:
        """
        Calculates E[sup(X)] for candidate itemsets across the UTD.
        E[sup(X)] = Sum over transactions of the Product of probabilities.
        """
        expected_support_counts = defaultdict(float)
        
        for transaction in transactions:
            transaction_items = set(transaction.keys())
            for candidate in candidates:
                if candidate.issubset(transaction_items):
                    joint_prob = 1.0
                    for item in candidate:
                        joint_prob *= transaction[item]
                    expected_support_counts[candidate] += joint_prob
                    
        frequent_candidates = {
            itemset: support 
            for itemset, support in expected_support_counts.items() 
            if support >= self.min_support
        }
        return frequent_candidates

    def fit(self, transactions: list):
        """
        Executes the exhaustive Probabilistic Apriori mining process.
        """
        logging.info("Executing Probabilistic Apriori Algorithm...")
        logging.info(f"Parameters -> min_sup: {self.min_support:.2f}, min_conf: {self.min_confidence:.2f}")
        k = 1
        
        c1 = set()
        for transaction in transactions:
            for item in transaction.keys():
                c1.add(frozenset([item]))
                
        l1 = self._calculate_expected_support(c1, transactions)
        
        if not l1:
            logging.warning("No frequent 1-itemsets discovered. Check support threshold.")
            return

        self.frequent_itemsets[k] = l1
        logging.info(f"Pass {k}: Discovered {len(l1)} frequent {k}-itemsets.")
        
        while True:
            k += 1
            candidates = self._generate_candidates(self.frequent_itemsets[k-1], k)
            
            if not candidates:
                break
                
            lk = self._calculate_expected_support(candidates, transactions)
            
            if not lk:
                break
                
            self.frequent_itemsets[k] = lk
            logging.info(f"Pass {k}: Discovered {len(lk)} frequent {k}-itemsets.")
            
    def generate_rules(self) -> list:
        """
        Generates robust probabilistic association rules utilizing Expected Confidence.
        """
        logging.info("Commencing Probabilistic Rule Generation...")
        rules = []
        
        for k in range(2, len(self.frequent_itemsets) + 1):
            for itemset, union_support in self.frequent_itemsets[k].items():
                
                for i in range(1, k):
                    for antecedent_tuple in combinations(itemset, i):
                        antecedent = frozenset(antecedent_tuple)
                        consequent = itemset - antecedent
                        
                        antecedent_support = self.frequent_itemsets[len(antecedent)].get(antecedent, 0)
                        
                        if antecedent_support == 0:
                            continue
                            
                        expected_confidence = union_support / antecedent_support
                        
                        if expected_confidence >= self.min_confidence:
                            consequent_support = self.frequent_itemsets[len(consequent)].get(consequent, 1)
                            lift = (union_support / (antecedent_support * consequent_support)) if antecedent_support * consequent_support > 0 else 0
                            
                            rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'expected_support': union_support,
                                'expected_confidence': expected_confidence,
                                'lift': lift
                            })
        return rules
