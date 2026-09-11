"""Writes an SVG color triangle: red at the top corner, green at the bottom left
and blue at the bottom right, linearly mixed in between."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SIDE    = 400.0                                                        # edge length, in SVG user units
CORNERS = {'top': '#ff0000', 'left': '#00ff00', 'right': '#0000ff'}   # position -> color
PATH    = Path('scripts/output/color_triangle.svg')


def main():
    height   = SIDE * np.sqrt(3) / 2
    vertices = np.array([[SIDE / 2, 0.0], [0.0, height], [SIDE, height]])  # top, left, right; y points down
    points   = ' '.join(f'{x:.3f},{y:.3f}' for x, y in vertices)

    svg = ET.Element('svg', {
        'xmlns':   'http://www.w3.org/2000/svg',
        'width':   f'{SIDE:.3f}',
        'height':  f'{height:.3f}',
        'viewBox': f'0 0 {SIDE:.3f} {height:.3f}',
    })
    defs  = ET.SubElement(svg, 'defs')
    group = ET.SubElement(svg, 'g', {'style': 'isolation:isolate'})

    for vertex, (name, color) in zip(vertices, CORNERS.items()):
        # Runs from the vertex to the foot of its altitude, so the gradient at a
        # point is the vertex color times that vertex's barycentric weight.
        a, b = vertices[~np.all(vertices == vertex, axis=1)]
        foot = a + np.dot(vertex - a, b - a) / np.dot(b - a, b - a) * (b - a)

        gradient = ET.SubElement(defs, 'linearGradient', {
            'id':            name,
            'gradientUnits': 'userSpaceOnUse',
            'x1': f'{vertex[0]:.3f}', 'y1': f'{vertex[1]:.3f}',
            'x2': f'{foot[0]:.3f}',   'y2': f'{foot[1]:.3f}',
        })
        ET.SubElement(gradient, 'stop', {'offset': '0', 'stop-color': color})
        ET.SubElement(gradient, 'stop', {'offset': '1', 'stop-color': '#000000'})

        # Screen blending adds colors whose channels do not overlap.
        ET.SubElement(group, 'polygon', {
            'points': points,
            'fill':   f'url(#{name})',
            'style':  'mix-blend-mode:screen',
        })

    ET.indent(svg)
    PATH.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(svg).write(PATH, encoding='utf-8', xml_declaration=True)
    print(f'wrote {PATH}')

if __name__ == '__main__':
    main()
