package ptithcm.service;

import java.util.List;

import javax.transaction.Transactional;

import org.springframework.stereotype.Service;

import ptithcm.entity.LoaiSanPhamEntity;
@Transactional
@Service
public interface loaiSanPhamService {
	public LoaiSanPhamEntity layLoaiTheoMa(String maLoai);
	public List<LoaiSanPhamEntity> layLoai();
	public List<LoaiSanPhamEntity> layLoaiTheoGioiTinh(String gioiTinh);
	boolean kiemTraLoai(String maLoai);
	
	public void themLoai(LoaiSanPhamEntity loai);
	public void updateLoai(LoaiSanPhamEntity loai);
	public void xoaLoai(LoaiSanPhamEntity loai);
	
}
