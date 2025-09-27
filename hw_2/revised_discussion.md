# Discussion

## Degree distribution

From the histograms, the **airport network** spans a much wider range of degrees than **EuroRoad**. A large share of airports have ≤10 connections, but there’s a long heavy tail with high-degree hubs—this is what you’d expect from a hub-and-spoke system. By contrast, **EuroRoad** is narrowly distributed with \(k_{\max}=10\) and many nodes of degree 2 (road segments between intersections).

The summary stats back this up:

- **EuroRoad:** \(N=1174\), \(E=1417\), density \(=0.0021\) (0.21%), \(̄k \approx 2.41\).  
- **OpenFlights:** \(N=2939\), \(E=15677\), density \(=0.0036\) (0.36%), \(̄k \approx 10.67\).

Note that the higher average degree in OpenFlights is due to many more edges *per node* (design + fewer physical constraints), not because the network is larger. These hubs also explain the much **smaller diameter** in the largest component: **14** for OpenFlights versus **62** for EuroRoad. In other words, even though both graphs are sparse, the airline graph still keeps typical paths short, while the road graph—constrained by geography—does not.

Also worth calling out the giant component sizes: OpenFlights’ LCC includes **≈98.8%** of airports (2905/2939), while EuroRoad’s includes **≈88.5%** of cities (1039/1174), another sign the airport network is more globally connected.

## Clustering

Clustering diverges sharply:

- **EuroRoad:** average clustering **1.67%** → very few triangles; most places have just a couple of ways in/out.  
- **OpenFlights:** average clustering **45.26%** → strong local triadic closure; airports in the same region commonly interconnect.

In the \(C(k)\) plot, lower-clustering nodes in OpenFlights tend to be large hubs that connect many otherwise unconnected regions—classic behavior where \(C\) often decreases with \(k\). For EuroRoad, most nodes have low clustering (degree-2 chains dominate), so finding very short routes is harder.

One nuance: clustering doesn’t tell us whether a node is a “bridge” (articulation point); we didn’t measure that here. What we *can* say is that hubs in OpenFlights substantially shorten paths, while EuroRoad’s low clustering plus planarity constraints keep the network stretched out.

## Assortativity

Your coefficients show:

- **EuroRoad:** \(r = 0.1267\) (more positively assortative).  
- **OpenFlights:** \(r = 0.0509\) (weakly positive / near-neutral).

So EuroRoad exhibits a stronger tendency for similar-degree nodes to connect (high-degree cities linking to other high-degree cities), which fits regional hierarchies in a road system. OpenFlights is only weakly assortative—consistent with many links between small airports and big hubs—so \(k\) and average neighbor degree correlate less tightly. This hub-spoke mixing helps keep path lengths short even for low-degree airports.
