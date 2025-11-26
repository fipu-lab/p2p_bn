from p2p.agents.p2p_agent import *
import numpy as np
import tensorflow as tf


class BertAgent(P2PAgent):
    def __init__(
        self,
        multi_task_optimization=True,
        multi_task_differentiation="model",
        multi_output_averaging=False,
        aggregated_bert_beta=0.9,  # EMA beta for aggregated bert
        tail_ema_beta=0.9,  # EMA beta for cached tails
        bert_blend_alpha=0.6,  # how much to keep personal bert vs aggregated (alpha*personal + (1-alpha)*agg)
        tail_beta=0.7,
        **kwargs
    ):
        assert multi_task_differentiation in ["model", "data"]
        if "early_stopping" not in kwargs:
            kwargs["early_stopping"] = False
        kwargs["data_pars"]["caching"] = True
        super(BertAgent, self).__init__(**kwargs)
        self.bert_layer = self.model.layers[3]
        self.output_layer = self.model.layers[-1]
        self.multi_task_optimization = multi_task_optimization
        self.multi_task_differentiation = multi_task_differentiation
        self.multi_output_averaging = multi_output_averaging
        self.tail_beta = tail_beta

        # New fields for aggregation
        # aggregated shared bert weights and a count for weighted averaging
        self._aggregated_bert = {
            "weights": self.bert_layer.get_weights(),
            "count": 1,
            "beta": aggregated_bert_beta,
        }

        # local cache maps tail_size -> {weights, count, beta}
        # existing agents' tails cached as aggregated per tail-size (task)
        self._local_cache = {}
        self._global_count = 1

        # thresholds / hyperparams
        self._tail_ema_beta = tail_ema_beta
        self._bert_blend_alpha = bert_blend_alpha

    @staticmethod
    def _diversity_aware_blend(w_self, w_other, alpha=0.5, gamma=0.05):
        """Blend weights while adding a small orthogonal perturbation (NumPy-safe)."""
        blended = []
        for a, b in zip(w_self, w_other):
            a_np, b_np = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
            diff = a_np - b_np
            # mean direction (scalar)
            mean_dir = np.mean(diff)
            # residual = orthogonal residual around mean
            residual = diff - mean_dir
            blended_w = alpha * a_np + (1 - alpha) * b_np + gamma * residual
            blended.append(blended_w)
        return blended

    # ---------- helper functions ----------
    @staticmethod
    def _flatten_weights(weights):
        flat = np.concatenate([w.reshape(-1) for w in tf.nest.flatten(weights)])
        return flat

    def _cosine_similarity(self, w1, w2, eps=1e-12):
        f1 = self._flatten_weights(w1)
        f2 = self._flatten_weights(w2)
        denom = (np.linalg.norm(f1) * np.linalg.norm(f2)) + eps
        return float(np.dot(f1, f2) / denom)

    @staticmethod
    def _weighted_average_weights(w_a, count_a, w_b, count_b):
        # Federated weighted average for nested weight structures
        total = count_a + count_b
        return tf.nest.map_structure(
            lambda a, b: (a * count_a + b * count_b) / total, w_a, w_b
        )

    @staticmethod
    def _ema_update(old_w, new_w, beta):
        # EMA update for nested weight structures
        return tf.nest.map_structure(
            lambda o, n: beta * o + (1.0 - beta) * n, old_w, new_w
        )

    # ---------- main receive_message ----------
    def receive_message(self, other_agent):
        # keep original semantics: call parent's receive_message properly
        super(P2PAgent, self).receive_message(other_agent)

        # Quick access
        my_units = self.output_layer.units
        other_units = other_agent.output_layer.units
        equal_tails = my_units == other_units

        different_tasks = (
            self.multi_task_differentiation == "model"
            and self.model.layers[-1].units != other_agent.model.layers[-1].units
            or self.multi_task_differentiation == "data"
            and self.dataset_name != other_agent.dataset_name
        )

        # If multi_output_averaging is requested, use the new aggregated approach
        if self.multi_output_averaging:
            # CASE A: exact same tail (same task / units)
            if not different_tasks:
                # If both have cached meta counts, attempt weighted federated averaging of full models
                my_count = self._global_count
                other_count = other_agent._global_count
                # Weighted average full model (federated)
                weights = self._weighted_average_weights(
                    self.get_model_weights(),
                    my_count,
                    other_agent.get_model_weights(),
                    other_count,
                )
                self.set_model_weights(weights)

                # update aggregated bert record too (increase count)
                self._aggregated_bert["weights"] = self._weighted_average_weights(
                    self._aggregated_bert["weights"],
                    self._aggregated_bert.get("count", 1),
                    other_agent.bert_layer.get_weights(),
                    other_count,
                )
                self._aggregated_bert["count"] = (
                    self._aggregated_bert.get("count", 1) + other_count
                )

                # If other has cached tails, merge them into our cache (per-tail weighted)
                if getattr(other_agent, "_local_cache", None):
                    for key, other_entry in other_agent._local_cache.items():
                        if key in self._local_cache:
                            our_entry = self._local_cache[key]
                            merged_weights = self._weighted_average_weights(
                                our_entry["weights"],
                                our_entry.get("count", 1),
                                other_entry["weights"],
                                other_entry.get("count", 1),
                            )
                            our_entry["weights"] = merged_weights
                            our_entry["count"] = our_entry.get(
                                "count", 1
                            ) + other_entry.get("count", 1)
                        else:
                            # copy other's cached tail
                            self._local_cache[key] = {
                                "weights": other_entry["weights"],
                                "count": other_entry.get("count", 1),
                                "beta": other_entry.get("beta", self._tail_ema_beta),
                            }

                # mark counts for potential future weighted merges
                self._global_count = self._global_count + other_agent._global_count

            # CASE B: tails differ -> perform shared-bert aggregation and per-tail EMA updates
            else:
                # --- Update aggregated shared BERT ---
                self._aggregated_bert["weights"] = self._ema_update(
                    self._aggregated_bert["weights"],
                    other_agent.bert_layer.get_weights(),
                    self._aggregated_bert.get("beta", self._bert_blend_alpha),
                )
                self._aggregated_bert["count"] = (
                    self._aggregated_bert.get("count", 1) + 1
                )

                # --- Update other agent's tail into local_cache (EMA-based) ---
                # If we don't have a cache entry for other_units, create it
                if other_units not in self._local_cache:
                    self._local_cache[other_units] = {
                        "weights": other_agent.output_layer.get_weights(),
                        "count": 1,
                        "beta": self._tail_ema_beta,
                    }
                else:
                    entry = self._local_cache[other_units]
                    # use weighted average by counts if both have counts, else EMA
                    if entry.get("count", 0) and other_agent._global_count:
                        merged = self._weighted_average_weights(
                            entry["weights"],
                            entry.get("count", 1),
                            other_agent.output_layer.get_weights(),
                            other_agent._global_count,
                        )
                        entry["weights"] = merged
                        entry["count"] = (
                            entry.get("count", 1) + other_agent._global_count
                        )
                    else:
                        entry["weights"] = self._ema_update(
                            entry["weights"],
                            other_agent.output_layer.get_weights(),
                            entry.get("beta", self._tail_ema_beta),
                        )
                        entry["count"] = entry.get("count", 1) + 1

                # If we already have a cached entry matching our own tail size in other_agent's cache (mutual),
                # merge and partially update local tail
                if my_units in other_agent._local_cache:
                    other_cached = other_agent._local_cache[my_units]
                    our_tail_weights = self.output_layer.get_weights()
                    # merge our tail with other cached tail using weighted average
                    merged_tail = self._weighted_average_weights(
                        our_tail_weights,
                        self._global_count,
                        other_cached["weights"],
                        other_cached.get("count", 1),
                    )
                    # set/update local tail but keep some personalization (blend)
                    blended_tail = self._ema_update(
                        our_tail_weights, merged_tail, self.tail_beta
                    )  # keep more of our own tail
                    self.model.layers[-1].set_weights(blended_tail)

                # Finally: apply aggregated bert to our current bert with blending (personalization/proximal)
                # Diversity-aware blend (instead of plain averaging)
                personal_bert = self.bert_layer.get_weights()
                agg_bert = self._aggregated_bert["weights"]
                blended = self._diversity_aware_blend(
                    personal_bert, agg_bert, alpha=self._bert_blend_alpha, gamma=0.02
                )
                self.bert_layer.set_weights(blended)

                if self.multi_task_optimization:
                    self.bert_layer.trainable = False

        else:
            if different_tasks:
                self.bert_layer.set_weights(
                    tf.nest.map_structure(
                        lambda a, b: (a + b) / 2.0,
                        self.bert_layer.get_weights(),
                        other_agent.bert_layer.get_weights(),
                    )
                )

            else:
                weights = tf.nest.map_structure(
                    lambda a, b: (a + b) / 2.0,
                    self.get_model_weights(),
                    other_agent.get_model_weights(),
                )
                self.set_model_weights(weights)

        self.received_msg = True
        self.train_rounds = 1

        return True

    def fit(self, epochs=0):
        super(BertAgent, self).fit(epochs)
        if self.multi_task_optimization:
            self.bert_layer.trainable = True
