package models

import (
	"sort"
)

var irqRelatedMetrics = []string{"kepler_container_bpf_block_irq_total", "kepler_container_bpf_net_tx_irq_total", "kepler_container_bpf_net_rx_irq_total", "kepler_container_bpf_cpu_time_us_total"}
var irqPowerLabel = []string{"kepler_node_package_joules_total"}

type DataLocation int64

const (
	Prometheus DataLocation = iota
	Local
)

type ModelFeatureType int64

const (
	irqFeatures ModelFeatureType = iota
)

type ModelLabelType int64

const (
	totalPackagePower ModelLabelType = iota
)

func sort_model_names(features []string) []string {
	copied_features := make([]string, len(features))
	copy(copied_features, features)
	sort.Strings(copied_features)
	return copied_features
}

var Model_Feature_Groups = map[ModelFeatureType][]string{
	irqFeatures: sort_model_names(irqRelatedMetrics),
}

var Model_Label_Groups = map[ModelLabelType][]string{
	totalPackagePower: sort_model_names(irqPowerLabel),
}
