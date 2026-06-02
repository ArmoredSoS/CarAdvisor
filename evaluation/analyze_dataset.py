#!/usr/bin/env python3
"""Dataset analysis script for CarAdvisor."""

from Dataset import cars
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset():
    """Analyze the car dataset and output statistics."""
    
    # Total number of entries
    total_entries = len(cars)
    
    # Fields to analyze distribution (excluding car_brand as they're all unique)
    distribution_fields = [
        "car_type", "fuel_type", "car_usecase", 
        "car_state", "car_dimensions", "fuel_efficiency", "car_design"
    ]
    
    # Calculate distributions
    distributions = {}
    for field in distribution_fields:
        distributions[field] = defaultdict(int)
        for car in cars:
            value = car.get(field)
            if value:
                distributions[field][value] += 1
    
    # Calculate average price overall
    total_price = sum(car["car_price"] for car in cars)
    average_price = total_price / total_entries if total_entries > 0 else 0
    
    # Calculate average price per car type
    car_types = defaultdict(list)
    for car in cars:
        car_type = car.get("car_type")
        if car_type:
            car_types[car_type].append(car["car_price"])
    
    average_price_by_type = {}
    for car_type, prices in car_types.items():
        average_price_by_type[car_type] = sum(prices) / len(prices) if prices else 0
    
    # Print results
    print("=" * 70)
    print("DATASET ANALYSIS REPORT")
    print("=" * 70)
    
    print(f"\n📊 TOTAL ENTRIES: {total_entries}")
    
    print(f"\n💰 OVERALL STATISTICS:")
    print(f"   Average Price: €{average_price:,.2f}")
    print(f"   Min Price: €{min(car['car_price'] for car in cars):,.2f}")
    print(f"   Max Price: €{max(car['car_price'] for car in cars):,.2f}")
    print(f"   Total Value: €{total_price:,.2f}")
    
    print(f"\n📈 DISTRIBUTION BY FIELD:")
    for field in distribution_fields:
        print(f"\n   {field.upper().replace('_', ' ')}:")
        dist = distributions[field]
        sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        for value, count in sorted_dist:
            percentage = (count / total_entries) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"      {value:.<20} {count:>3} ({percentage:>5.1f}%) {bar}")
    
    print(f"\n💵 AVERAGE PRICE BY CAR TYPE:")
    sorted_types = sorted(average_price_by_type.items(), key=lambda x: x[1])
    for car_type, avg_price in sorted_types:
        count = len(car_types[car_type])
        print(f"   {car_type:.<20} €{avg_price:>12,.2f} (n={count})")
    
    print("\n" + "=" * 70)
    
    # Create visualizations
    create_visualizations(distributions, average_price_by_type, total_entries, average_price)
    
    # Export detailed statistics to JSON
    export_data = {
        "total_entries": total_entries,
        "overall_statistics": {
            "average_price": round(average_price, 2),
            "min_price": min(car["car_price"] for car in cars),
            "max_price": max(car["car_price"] for car in cars),
            "total_value": round(total_price, 2)
        },
        "distributions": {
            field: dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))
            for field, dist in distributions.items()
        },
        "average_price_by_car_type": {
            car_type: round(avg_price, 2)
            for car_type, avg_price in average_price_by_type.items()
        }
    }
    
    with open("dataset_analysis.json", "w") as f:
        json.dump(export_data, f, indent=2)
    print("✅ Detailed statistics exported to dataset_analysis.json")
    print("📊 Graphs saved to dataset_analysis_charts.png\n")

def create_visualizations(distributions, average_price_by_type, total_entries, average_price):
    """Create and save visualization charts."""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Car Dataset Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    # 1. Car Type Distribution
    ax1 = plt.subplot(3, 3, 1)
    car_type_dist = distributions["car_type"]
    types = sorted(car_type_dist.keys(), key=lambda x: car_type_dist[x], reverse=True)
    counts = [car_type_dist[t] for t in types]
    colors = plt.cm.Set3(np.linspace(0, 1, len(types)))
    ax1.bar(types, counts, color=colors)
    ax1.set_title('Car Type Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
    
    # 2. Fuel Type Distribution
    ax2 = plt.subplot(3, 3, 2)
    fuel_type_dist = distributions["fuel_type"]
    fuels = sorted(fuel_type_dist.keys(), key=lambda x: fuel_type_dist[x], reverse=True)
    fuel_counts = [fuel_type_dist[f] for f in fuels]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax2.pie(fuel_counts, labels=fuels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Fuel Type Distribution', fontweight='bold')
    
    # 3. Car Usecase Distribution
    ax3 = plt.subplot(3, 3, 3)
    usecase_dist = distributions["car_usecase"]
    usecases = sorted(usecase_dist.keys(), key=lambda x: usecase_dist[x], reverse=True)
    usecase_counts = [usecase_dist[u] for u in usecases]
    colors = plt.cm.Set2(np.linspace(0, 1, len(usecases)))
    ax3.barh(usecases, usecase_counts, color=colors)
    ax3.set_title('Car Usecase Distribution', fontweight='bold')
    ax3.set_xlabel('Count')
    for i, v in enumerate(usecase_counts):
        ax3.text(v + 0.5, i, str(v), va='center', fontsize=9)
    
    # 4. Car State Distribution
    ax4 = plt.subplot(3, 3, 4)
    state_dist = distributions["car_state"]
    states = sorted(state_dist.keys(), key=lambda x: state_dist[x], reverse=True)
    state_counts = [state_dist[s] for s in states]
    colors = ['#90EE90', '#FFB6C1']
    ax4.pie(state_counts, labels=states, autopct='%1.1f%%', colors=colors, startangle=90)
    ax4.set_title('Car State Distribution', fontweight='bold')
    
    # 5. Car Dimensions Distribution
    ax5 = plt.subplot(3, 3, 5)
    dim_dist = distributions["car_dimensions"]
    dims = sorted(dim_dist.keys(), key=lambda x: dim_dist[x], reverse=True)
    dim_counts = [dim_dist[d] for d in dims]
    colors = plt.cm.Spectral(np.linspace(0, 1, len(dims)))
    ax5.bar(dims, dim_counts, color=colors)
    ax5.set_title('Car Dimensions Distribution', fontweight='bold')
    ax5.set_ylabel('Count')
    ax5.tick_params(axis='x', rotation=45)
    for i, v in enumerate(dim_counts):
        ax5.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
    
    # 6. Fuel Efficiency Distribution
    ax6 = plt.subplot(3, 3, 6)
    eff_dist = distributions["fuel_efficiency"]
    effs = ['high', 'medium', 'low']
    eff_counts = [eff_dist.get(e, 0) for e in effs]
    colors = ['#2ECC71', '#F39C12', '#E74C3C']
    ax6.bar(effs, eff_counts, color=colors)
    ax6.set_title('Fuel Efficiency Distribution', fontweight='bold')
    ax6.set_ylabel('Count')
    for i, v in enumerate(eff_counts):
        if v > 0:
            ax6.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
    
    # 7. Car Design Distribution
    ax7 = plt.subplot(3, 3, 7)
    design_dist = distributions["car_design"]
    designs = sorted(design_dist.keys(), key=lambda x: design_dist[x], reverse=True)
    design_counts = [design_dist[d] for d in designs]
    colors = ['#9B59B6', '#3498DB', '#E67E22']
    ax7.bar(designs, design_counts, color=colors)
    ax7.set_title('Car Design Distribution', fontweight='bold')
    ax7.set_ylabel('Count')
    for i, v in enumerate(design_counts):
        ax7.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
    
    # 8. Average Price by Car Type
    ax8 = plt.subplot(3, 3, 8)
    types_sorted = sorted(average_price_by_type.items(), key=lambda x: x[1])
    type_names = [t[0] for t in types_sorted]
    type_prices = [t[1] for t in types_sorted]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(type_names)))
    bars = ax8.barh(type_names, type_prices, color=colors)
    ax8.set_title('Average Price by Car Type', fontweight='bold')
    ax8.set_xlabel('Average Price (€)')
    for i, v in enumerate(type_prices):
        ax8.text(v + 5000, i, f'€{v:,.0f}', va='center', fontsize=9)
    
    # 9. Price Statistics Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
    PRICE STATISTICS SUMMARY
    
    Total Entries: {total_entries}
    
    Average Price: €{average_price:,.2f}
    
    Price Range: €10,000 - €600,000
    
    Most Common Type: 
    {max(distributions['car_type'].items(), key=lambda x: x[1])[0]}
    
    Most Common Fuel Type:
    {max(distributions['fuel_type'].items(), key=lambda x: x[1])[0]}
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', 
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('dataset_analysis_charts.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    analyze_dataset()

