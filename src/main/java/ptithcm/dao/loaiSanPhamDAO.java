package ptithcm.dao;

import java.util.List;

import ptithcm.entity.LoaiSanPhamEntity;
import ptithcm.entity.LoaiSanPhamEntity;

public interface loaiSanPhamDAO {
	public LoaiSanPhamEntity layLoaiTheoMa(String maLoai);
	public List<LoaiSanPhamEntity> layLoai();
	public List<LoaiSanPhamEntity> layLoaiTheoGioiTinh(String gioiTinh);
	
	public void themLoai(LoaiSanPhamEntity loai);
	public void updateLoai(LoaiSanPhamEntity loai);
	public void xoaLoai(LoaiSanPhamEntity loai);
}
