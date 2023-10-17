import math
import typing as t

import torch


def roll_polygon(vert: torch.Tensor) -> torch.Tensor:
    """Roll the vertices of a polygon to get consecutive edges.

    Args:
        vert (torch.Tensor): A tensor of shape (..., n, 2) where the last
            dimension contains the x and y coordinates of the vertices of the polygon.

    Returns:
        torch.Tensor: A tensor of shape (..., n, 2, 2) where the last two
            dimensions contain the coordinates of the edges of the polygon.
    """
    return torch.stack([vert, vert.roll(1, dims=-2)], dim=-1)


def intersect_segments(
    line_a: torch.Tensor, line_b: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Intersect two batches of edges. The intersection is computed using the segment
    intersection algorithm: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection

    Args:
        line_a (torch.Tensor): A tensor of shape (..., n, 2, 2) containing the
            coordinates of the edges of the first batch of line segments.
        line_b (torch.Tensor): Same as line_a, but for the second batch of line
            segments. The two batches must have the same shape.

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors of shape
            (..., n, m, 2) and (..., n, m) respectively. The first tensor contains the
            coordinates of the intersection points of the edges, and the second tensor
            contains a mask indicating which intersections are valid.
    """
    d12 = line_a[..., 0] - line_a[..., 1]
    d13 = line_a[..., 0] - line_b[..., 0]
    d34 = line_b[..., 0] - line_b[..., 1]
    nt = d13[..., 0] * d34[..., 1] - d13[..., 1] * d34[..., 0]
    nu = d13[..., 0] * d12[..., 1] - d13[..., 1] * d12[..., 0]
    dn = d12[..., 0] * d34[..., 1] - d12[..., 1] * d34[..., 0]
    t, u = torch.where(dn == 0.0, -1.0, nt / dn), torch.where(dn == 0.0, -1.0, nu / dn)
    mask = (t > 0) * (t < 1) * (u > 0) * (u < 1)
    ints = line_a[..., 0] + t.unsqueeze(-1) * (line_a[..., 1] - line_a[..., 0])
    ints = ints * mask.unsqueeze(-1).float()
    return ints, mask


def intersect_polygons(
    vert_a: torch.Tensor, vert_b: torch.Tensor
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    """Intersect all edges of two batches of polygons.

    Args:
        vert_a (torch.Tensor): A tensor of shape (..., n, 2) where the last
            dimension contains the x and y coordinates of the vertices of the first
            batch of polygons.
        vert_b (torch.Tensor): Same as vert_a, but for the second batch of polygons.
            The two batches must have the same shape.

    Returns:
        t.Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors of shape
            (..., n * m, 2) and (..., n * m) respectively. The first tensor contains
            the coordinates of the intersection points of the edges, and the second
            tensor contains a mask indicating which intersections are valid.
    """
    edge_a, edge_b = roll_polygon(vert_a), roll_polygon(vert_b)
    batchdims = [1] * (len(edge_a.shape) - 3)
    na, nb = edge_a.shape[-3], edge_b.shape[-3]
    edge_a_ = edge_a.unsqueeze(-4).repeat(batchdims + [nb, 1, 1, 1])
    edge_b_ = edge_b.unsqueeze(-3).repeat(batchdims + [1, na, 1, 1])
    ints, mask = intersect_segments(edge_a_, edge_b_)
    ints = ints.view(edge_a.shape[:-3] + (na * nb, 2))
    mask = mask.view(edge_a.shape[:-3] + (na * nb,))
    return ints, mask


def _compute_area(vert: torch.Tensor) -> torch.Tensor:
    """Compute the area of a batch of polygons.

    Args:
        vert (torch.Tensor): A tensor of shape (..., n, 2) where the last dimension
            contains the x and y coordinates of the vertices of the polygons.

    Returns:
        torch.Tensor: A tensor of shape (...) containing the area of the polygons.
    """
    res = vert[..., :-1, 0] * vert[..., 1:, 1] - vert[..., :-1, 1] * vert[..., 1:, 0]
    return res.sum(-1).abs() / 2


def _p_in_rect(points: torch.Tensor, rect: torch.Tensor) -> torch.Tensor:
    """Return a mask indicating which points are inside a batch of rectangles.

    Args:
        points (torch.Tensor): A tensor of shape (..., n, 2) where the last dimension
            contains the x and y coordinates of the points.
        rect (torch.Tensor): A tensor of shape (..., 4, 2) where the last dimension
            contains the x and y coordinates of the vertices of the rectangles. This
            is required to be a batch of rectangles, the function will not work with
            other polygons.

    Returns:
        torch.Tensor: A tensor of shape (..., n) containing a mask indicating which
            points are inside the rectangles.
    """
    a, b, d = rect[..., 0:1, :], rect[..., 1:2, :], rect[..., 3:4, :]
    ab, ad, ap = b - a, d - a, points - a
    p_ab = torch.sum(ab * ap, dim=-1) / torch.sum(ab * ab, dim=-1)
    p_ad = torch.sum(ad * ap, dim=-1) / torch.sum(ad * ad, dim=-1)
    cond1 = (p_ab > -1e-6) * (p_ab < 1 + 1e-6)
    cond2 = (p_ad > -1e-6) * (p_ad < 1 + 1e-6)
    return cond1 * cond2


def _sort_vertices(vert: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Sorts the vertices of a batch of octagons so that the vertices are in order
    around the center of the octagon, and the first vertex is the one with the
    smallest angle wrt the x axis. The invalid vertices are sorted to the end of the
    tensor and padded with the first vertex of the polygon, making all the polygons
    octagons with some degenerate edges.

    Args:
        vert (torch.Tensor): A tensor of shape (..., n, 2) where the last dimension
            contains the x and y coordinates of the vertices of the polygons.
        mask (torch.Tensor): A tensor of shape (..., n) containing a mask indicating
            which vertices are valid.

    Returns:
        torch.Tensor: A tensor of shape (..., 9, 2) containing the coordinates of the
            vertices of the polygons, sorted as described above.
    """
    # Gather info
    dev = vert.device
    batchdims = [1] * (len(vert.shape) - 2)

    # Number of valid vertices per polygon
    num_valid = torch.sum(mask.int(), dim=-1).long()

    # Normalize the vertices
    mean = torch.sum(
        vert * mask.float().unsqueeze(-1), dim=-2, keepdim=True
    ) / num_valid.unsqueeze(-1).unsqueeze(-1)
    norm_vert = vert - mean

    # Score = angle
    angles = torch.atan2(norm_vert[..., 1], norm_vert[..., 0]) + torch.pi

    # Angles with mask zero must be set to > 2pi, so they are sorted to the end
    angles = angles * mask.float() + (2 * torch.pi + 1) * (1 - mask.float())

    # Sort the angles
    _, indices = torch.sort(angles, dim=-1)

    # Just before the padding starts, we need to insert the indices of the first vertex
    # (which is the same as the last vertex)
    firsts = indices[..., 0:1]

    # Keep only the first 9 indices - no need to sort more than 9 vertices
    indices = indices[..., :9]

    # Index mask
    idxmask = torch.arange(9, device=dev).reshape(*batchdims, 9)
    idxmask = idxmask < num_valid.unsqueeze(-1)

    # Apply the mask
    indices = torch.where(idxmask, indices, firsts)

    # Gather the vertices
    indices = indices.long().unsqueeze(-1).repeat([*batchdims, 1, 2])
    return torch.gather(vert, -2, indices)


def rect_intersection(vert_a: torch.Tensor, vert_b: torch.Tensor) -> torch.Tensor:
    """Intersect two batches of rectangles, retrieving a batch of polygons as a result.
    As two rectangles can intersect in a polygon with up to 8 vertices, the result is
    padded with zeros to have 9 vertices per polygon, the last vertex being the same
    as the first one.

    In case of polygons with less than 9 vertices, the padding is done with the first
    vertex of the polygon, essentially making all the polygons octagons with degenerate
    edges. This is done to allow for a fully vectorized implementation of the
    intersection algorithm.

    Args:
        vert_a (torch.Tensor): A tensor of shape (..., 4, 2) where the last dimension
            contains the x and y coordinates of the vertices of the first batch of
            rectangles.
        vert_b (torch.Tensor): Same as vert_a, but for the second batch of rectangles.
            The two batches must have the same shape.

    Returns:
        torch.Tensor: A tensor of shape (..., 9, 2) containing the coordinates of the
            vertices of the polygons resulting from the intersection of the rectangles.
    """
    vert_int, mask = intersect_polygons(vert_a, vert_b)
    a_in_b, b_in_a = _p_in_rect(vert_a, vert_b), _p_in_rect(vert_b, vert_a)
    vertices = torch.cat([vert_a, vert_b, vert_int], dim=-2)
    mask = torch.cat([a_in_b, b_in_a, mask], dim=-1)
    return _sort_vertices(vertices, mask)


def get_rect_vert(rect: torch.Tensor) -> torch.Tensor:
    """Get the vertices of a rectangle.

    Args:
        rect (torch.Tensor): A tensor of shape (..., 5) where the last dimension
            contains the x, y, width, height and angle of the rectangle. Angles are
            in radians, wrt the x axis, counterclockwise.

    Returns:
        torch.Tensor: A tensor of shape (..., 4, 2) where the last dimension contains
            the x and y coordinates of the vertices of the rectangle.
    """
    # Device
    dev = rect.device

    # Initialize the vertices
    vert = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], device=dev) / 2

    # Match the batch size(s)
    nrect = math.prod(rect.shape[:-1])
    vert = vert.unsqueeze(0).repeat([nrect, 1, 1]).view(rect.shape[:-1] + (4, 2))

    # Scale the vertices
    vert = vert * rect[..., 2:4].unsqueeze(-2)

    # Rotate the vertices
    a = rect[..., 4]
    sin, cos = torch.sin(a), torch.cos(a)
    rot = torch.stack([cos, -sin, sin, cos], dim=-1).view(rect.shape[:-1] + (2, 2))
    vert = vert @ rot

    # Translate the vertices
    vert = vert + rect[..., 0:2].unsqueeze(-2)

    return vert


def compute_iou(rect_a: torch.Tensor, rect_b: torch.Tensor) -> torch.Tensor:
    """Compute the intersection over union between two bathes of rectangles.

    Args:
        rect_a (torch.Tensor): A tensor of shape (..., 5) where the last dimension
            contains the x, y, width, height and angle of the rectangle. Angles are
            in radians, wrt the x axis, counterclockwise.
        rect_b (torch.Tensor): Same as rect_a, but for the second batch of rectangles.
            The two batches must have the same shape.

    Returns:
        torch.Tensor: A tensor of shape (...) containing the intersection over union
            between the two batches of rectangles.
    """
    vert_a, vert_b = get_rect_vert(rect_a), get_rect_vert(rect_b)
    inter = rect_intersection(vert_a, vert_b)
    area = _compute_area(inter)
    area_a = rect_a[..., 2] * rect_a[..., 3]
    area_b = rect_b[..., 2] * rect_b[..., 3]
    return area / (area_a + area_b - area)
